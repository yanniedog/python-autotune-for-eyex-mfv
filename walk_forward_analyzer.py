#!/usr/bin/env python3
"""
Adaptive Walk‑Forward Analyzer – walk_forward_analyzer.py
========================================================
Analyses a pre‑staged OHLCV CSV (as created by *pytune_mfv.py*) and returns a
single JSON summary of the best‑performing MFV period combination discovered
using a **multi‑stage adaptive search** with strict out‑of‑sample validation.

Highlights
----------
* **Three‑block timeline split** (benchmark, tuning, lock‑box).
* **Stage‑1 Latin‑hypercube seeds**, **Bayesian optimisation** thereafter.
* **Rolling‑origin cross‑validation** on the tuning block (purged gaps).
* **Lock‑box approval** – each candidate tested once on final 20 % of data.
* **Vectorised MFV/z‑score calculation** with a global cache.
* **Numba‑accelerated trade engine** for realistic SL/TP exits.
* **Adaptive early‑pruning** via Optuna MedianPruner.

Usage
-----
```bash
python walk_forward_analyzer.py temp_data_SOLUSDT_1s.csv SOLUSDT 1s
```
The script prints exactly one JSON blob to STDOUT and logs progress to STDERR.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numba import njit, prange  # type: ignore
import os

# Optional dependencies – only needed for stage‑2 optimisation
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:  # graceful fallback: dummy Optuna shim
    optuna = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MAX_PERIOD = 1200
MIN_PERIOD = 10
TOP_SEED_FRACTION = 0.2  # keep top‑20 % seeds for Bayesian stage
LATIN_HYPER_SEEDS = 128  # stage‑1 seed count
TUNING_FOLDS = 5
LOCKBOX_SHARE = 0.20
TUNING_SHARE = 0.20  # remaining 60 % is benchmark
SL_PCT = 0.015  # 1.5 % stop‑loss
TP_PCT = 0.03   # 3 % take‑profit

# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def _ppy(interval: str) -> int:
    """Approx periods per year given Binance interval string."""
    unit = interval[-1]
    qty = int(interval[:-1])
    if unit == "s":
        return int(31536000 / qty)
    if unit == "m":
        return int(525600 / qty)
    if unit == "h":
        return int(8760 / qty)
    if unit == "d":
        return int(365 / qty)
    raise ValueError(f"Unsupported interval {interval}")


def _rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum using numpy cumulative sum."""
    c = np.cumsum(np.insert(a, 0, 0))
    return c[window:] - c[:-window]


def _youden_j(tp: int, fp: int, tn: int, fn: int) -> float:
    recall = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return recall + spec - 1.0


# ──────────────────────────────────────────────────────────────
# Numba‑optimised trade engine
# ──────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def _bt(close: np.ndarray, signal: np.ndarray) -> Tuple[float, float, float]:
    """Back‑tests composite signal; returns (win_rate, max_dd_pct, ann_return)."""
    equity = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    for i in range(len(signal)):
        if signal[i] == 0:
            continue
        entry = close[i]
        direction = 1 if signal[i] > 0 else -1
        # iterate forward until exit – simplistic horizon = 50 bars or SL/TP
        exit_price = entry
        for j in range(i + 1, min(i + 50, len(close))):
            move = (close[j] - entry) / entry * direction
            if move >= TP_PCT:
                exit_price = entry * (1 + TP_PCT * direction)
                break
            if move <= -SL_PCT:
                exit_price = entry * (1 - SL_PCT * direction)
                break
            exit_price = close[j]
        ret = (exit_price - entry) / entry * direction
        equity *= 1 + ret
        peak = max(peak, equity)
        trades += 1
        if ret > 0:
            wins += 1
    win_rate = wins / trades if trades else 0.0
    max_dd = (peak - equity) / peak if peak else 0.0
    periods_per_year = len(close) / 252  # rough daily bars for ann.
    ann_ret = equity ** (1 / periods_per_year) - 1 if periods_per_year else 0.0
    return win_rate, max_dd, ann_ret


# ──────────────────────────────────────────────────────────────
# MFV + z‑score utilities (with cache)
# ──────────────────────────────────────────────────────────────

_zcache: Dict[int, np.ndarray] = {}


def _mfv_z(vol: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Rolling money‑flow volume z‑score for a given period (cached)."""
    if len(vol) < period * 2:
        logger.warning(f"Insufficient data for period {period}: only {len(vol)} bars.")
        return np.zeros(len(vol))
    key = period
    if key in _zcache:
        z = _zcache[key]
    else:
        mfv = vol * np.sign(np.diff(np.concatenate(([close[0]], close))))
        roll = _rolling_sum(mfv, period)
        mean = pd.Series(roll).rolling(period).mean().to_numpy()
        std = pd.Series(roll).rolling(period).std(ddof=0).to_numpy()
        z = (roll - mean) / (std + 1e-12)
        z = np.concatenate((np.zeros(period * 2 - 1), z))  # pad to same length
        _zcache[key] = z
    # Ensure output is same length as close
    if len(z) > len(close):
        z = z[-len(close):]
    elif len(z) < len(close):
        z = np.concatenate((np.zeros(len(close) - len(z)), z))
    return z


# ──────────────────────────────────────────────────────────────
# Dataset split helpers
# ──────────────────────────────────────────────────────────────

def _split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    lock_n = int(n * LOCKBOX_SHARE)
    tuning_n = int(n * TUNING_SHARE)
    benchmark = df.iloc[: n - tuning_n - lock_n]
    tuning = df.iloc[n - tuning_n - lock_n : n - lock_n]
    lock = df.iloc[n - lock_n :]
    return benchmark.reset_index(drop=True), tuning.reset_index(drop=True), lock.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────

def _score_metrics(j: float, calmar: float, max_dd: float, periods: tuple = None) -> float:
    # Add a penalty for clustered periods (if provided)
    penalty = 0.0
    if periods is not None:
        diffs = [abs(periods[0] - periods[1]), abs(periods[1] - periods[2]), abs(periods[0] - periods[2])]
        min_diff = min(diffs)
        # Penalize if any two periods are closer than 10% of the range
        range_ = MAX_PERIOD - MIN_PERIOD
        if min_diff < 0.1 * range_:
            penalty = -0.2 * (0.1 * range_ - min_diff) / (0.1 * range_)
    return 0.5 * j + 0.3 * calmar - 0.2 * max_dd + penalty


# ──────────────────────────────────────────────────────────────
# Core evaluators
# ──────────────────────────────────────────────────────────────

def _evaluate_combo(periods: Tuple[int, int, int], df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Returns (YoudenJ, calmar, max_dd, win_rate)."""
    if len(df) < max(periods) * 2:
        logger.warning(f"Insufficient data for periods {periods}: only {len(df)} bars.")
        return 0.0, 0.0, 0.0, 0.0
    vol = df["volume"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    z1 = _mfv_z(vol, close, periods[0])
    z2 = _mfv_z(vol, close, periods[1])
    z3 = _mfv_z(vol, close, periods[2])
    minlen = min(len(z1), len(z2), len(z3), len(close))
    z1, z2, z3, close = z1[-minlen:], z2[-minlen:], z3[-minlen:], close[-minlen:]
    sig = z1 + z2 + z3
    signal = np.sign(sig)

    # Classification counts for J on next‑bar direction
    dir1 = np.sign(np.diff(close, prepend=close[0]))
    tp = np.sum((signal > 0) & (dir1 > 0))
    tn = np.sum((signal < 0) & (dir1 < 0))
    fp = np.sum((signal > 0) & (dir1 < 0))
    fn = np.sum((signal < 0) & (dir1 > 0))
    j = _youden_j(tp, fp, tn, fn)

    win_rate, max_dd, ann_ret = _bt(close, signal)
    calmar = ann_ret / (max_dd + 1e-12) if max_dd else 0.0
    return j, calmar, max_dd, win_rate


# ──────────────────────────────────────────────────────────────
# Stage‑1: Latin‑hypercube seeds
# ──────────────────────────────────────────────────────────────

def _lhs_samples(n: int, dim: int, low: int, high: int) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((dim, n)) * (cut[1:] - cut[:-1]) + cut[:-1]
    lhs = np.transpose(u)
    rng.shuffle(lhs)
    periods = []
    for row in lhs:
        p = tuple(sorted((int(low + r * (high - low)) for r in row)))  # type: ignore
        if p[0] < p[1] < p[2]:
            periods.append(p)  # type: ignore
    return periods[:n]


# ──────────────────────────────────────────────────────────────
# Cross‑validation on tuning block
# ──────────────────────────────────────────────────────────────

def _cv_eval(periods: Tuple[int, int, int], df: pd.DataFrame) -> Tuple[float, float, float]:
    n = len(df)
    gap = max(periods)
    fold_size = n // TUNING_FOLDS
    if fold_size <= 0 or n < gap * 2:
        logger.warning(f"Insufficient data for cross-validation: {n} rows, gap {gap}, folds {TUNING_FOLDS}.")
        return 0.0, 0.0, 0.0
    js, calmars, dds = [], [], []
    for i in range(TUNING_FOLDS):
        start = i * fold_size
        end = start + fold_size
        train_idx = list(range(0, start - gap)) + list(range(end + gap, n))
        test_idx = list(range(start, end))
        if min(test_idx) < 0 or max(train_idx) >= n:
            continue  # skip if indices out of range
        test_df = df.iloc[test_idx]
        j, calmar, dd, _ = _evaluate_combo(periods, test_df)
        js.append(j)
        calmars.append(calmar)
        dds.append(dd)
    if not js:
        logger.warning(f"No valid folds for cross-validation: {n} rows, gap {gap}, folds {TUNING_FOLDS}.")
        return 0.0, 0.0, 0.0
    return float(np.mean(js)), float(np.mean(calmars)), float(np.mean(dds))


# ──────────────────────────────────────────────────────────────
# Main optimisation routine
# ──────────────────────────────────────────────────────────────

def optimise(df_bench: pd.DataFrame, df_tune: pd.DataFrame, symbol: str, interval: str) -> Tuple[Tuple[int, int, int], Dict[str, float]]:
    """Return best periods and metrics from tuning block. Persist and reuse Optuna study. Use adaptive search space if provided."""
    # Adaptive search space
    min_period = MIN_PERIOD
    max_period = MAX_PERIOD
    adaptive_json = os.environ.get('ADAPTIVE_SEARCH_SPACE', '')
    if adaptive_json:
        try:
            space = json.loads(adaptive_json)
            min_period = max(MIN_PERIOD, int(space.get('min', MIN_PERIOD)))
            max_period = min(MAX_PERIOD, int(space.get('max', MAX_PERIOD)))
            logger.info(f"Using adaptive search space: min={min_period}, max={max_period}")
        except Exception as e:
            logger.warning(f"Failed to parse adaptive search space: {e}")
    lhs = _lhs_samples(LATIN_HYPER_SEEDS, 3, min_period, max_period)
    logger.info("Stage‑1 LHS seeds: %d", len(lhs))
    seed_results = []
    for p in lhs:
        j, c, dd, w = _evaluate_combo(p, df_bench)
        seed_results.append((p, _score_metrics(j, c, dd, p)))
    seed_results.sort(key=lambda x: x[1], reverse=True)
    top_seeds = [p for p, _ in seed_results[: max(1, int(len(seed_results) * TOP_SEED_FRACTION))]]
    if optuna is None:
        logger.warning("Optuna not available – returning best LHS seed only")
        best_p = top_seeds[0]
        j, c, dd, w = _evaluate_combo(best_p, df_tune)
        return best_p, {"youden_j": j, "calmar": c, "max_dd": dd, "win_rate": w}
    # Persist study per symbol/interval
    study_name = f"study_{symbol}_{interval}"
    storage_path = f"sqlite:///optuna_{symbol}_{interval}.db"
    if os.path.exists(f"optuna_{symbol}_{interval}.db"):
        logger.info(f"Loading existing Optuna study for {symbol} {interval}")
        study = optuna.load_study(study_name=study_name, storage=storage_path)
    else:
        logger.info(f"Creating new Optuna study for {symbol} {interval}")
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=0),
            storage=storage_path,
            load_if_exists=True,
        )
    def objective(trial):
        p1 = trial.suggest_int("p1", min_period, max_period)
        p2 = trial.suggest_int("p2", p1 + 1, max_period)
        p3 = trial.suggest_int("p3", p2 + 1, max_period)
        periods = (p1, p2, p3)
        j_b, c_b, dd_b, _ = _evaluate_combo(periods, df_bench)
        score = -_score_metrics(j_b, c_b, dd_b, periods)
        trial.report(j_b, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        j, c, dd = _cv_eval(periods, df_tune)
        score += -_score_metrics(j, c, dd, periods)
        return score
    # Enqueue previous bests
    if len(study.trials) > 0:
        best_params = study.best_trial.params
        best_p = (best_params["p1"], best_params["p2"], best_params["p3"])
        study.enqueue_trial({"p1": best_p[0], "p2": best_p[1], "p3": best_p[2]})
    for p in top_seeds:
        study.enqueue_trial({"p1": p[0], "p2": p[1], "p3": p[2]})
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    best = study.best_trial.params
    best_p = (best["p1"], best["p2"], best["p3"])
    j, c, dd, w = _evaluate_combo(best_p, df_tune)
    logger.info(f"Best periods: {best_p}, spreads: {[abs(best_p[0]-best_p[1]), abs(best_p[1]-best_p[2]), abs(best_p[0]-best_p[2])]}")
    return best_p, {"youden_j": j, "calmar": c, "max_dd": dd, "win_rate": w}


# ──────────────────────────────────────────────────────────────
# Lock‑box validation
# ──────────────────────────────────────────────────────────────

def lockbox_validate(periods: Tuple[int, int, int], df_lock: pd.DataFrame) -> Dict[str, float]:
    j, c, dd, w = _evaluate_combo(periods, df_lock)
    return {"youden_j": j, "calmar": c, "max_dd": dd, "win_rate": w}


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def _main(path: Path, symbol: str, interval: str):
    try:
        t0 = time.time()
        df = pd.read_csv(path)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"CSV missing column {col}")
        if len(df) < MIN_PERIOD * 2:
            msg = f"Insufficient data: only {len(df)} rows, need at least {MIN_PERIOD * 2}."
            logger.error(msg)
            output = {"status": "fail", "error": msg}
            json.dump(output, sys.stdout, separators=(",", ":"))
            return
        df_bench, df_tune, df_lock = _split_dataset(df)
        best_p, tune_metrics = optimise(df_bench, df_tune, symbol, interval)
        lock_metrics = lockbox_validate(best_p, df_lock)
        result = {
            "status": "success",
            "symbol": symbol,
            "interval": interval,
            "periods": best_p,
            "tuning": tune_metrics,
            "lockbox": lock_metrics,
            "runtime_sec": round(time.time() - t0, 3),
        }
        output = {"data": result, "status": "success"}
        json.dump(output, sys.stdout, separators=(",", ":"))
    except Exception as e:
        logging.error(f'Exception in analyzer: {e}')
        logging.error(traceback.format_exc())
        output = {"status": "fail", "error": str(e)}
        json.dump(output, sys.stdout, separators=(",", ":"))
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error('Usage: walk_forward_analyzer.py <csv_path> <symbol> <interval>')
        sys.exit(1)
    _main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
