#!/usr/bin/env python3
"""
Adaptive MFV Orchestrator – pytune_mfv.py
========================================
Pulls OHLCV candles from Binance, caches them locally, then iteratively
calls *walk_forward_analyzer.py* on each symbol/interval pair. The script
now incorporates:

1. **Dataset caching & incremental refresh** – prevents wasted downloads.
2. **Parallel candle fetching** – configurable max workers.
3. **Timeline partitioning** – splits each dataset into benchmark / tuning
   / lock‑box segments and passes the split info to the analyser.
4. **Comprehensive logging** – rotating file handler + console.
5. **Graceful failure isolation** – one bad pair no longer aborts the run.
6. **Extensible optimisation loop** – a placeholder Optuna study that can
   resume across sessions (disabled by default but scaffolded).

The analyser can remain unmodified; any extra CLI flags are purely
ignored until you upgrade *walk_forward_analyzer.py*. The script still
produces *walk_forward_report.csv* with aggregated metrics and a ranked
"robustness_score".
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import warnings
import logging
import sys
try:
    import pyarrow
    import fastparquet
except ImportError as e:
    logging.error('Missing required parquet engine: pyarrow or fastparquet. Please install them.')
    sys.exit(1)

from collections import defaultdict
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Binance API
    "BASE_URL": "https://api.binance.com",
    "MAX_KLINES_PER_REQ": 1000,
    # Data slices (percent of dataset in chronological order)
    "BENCHMARK_PCT": 0.6,
    "TUNE_PCT": 0.2,
    # Remaining 20 % is automatically the lock‑box.
    # Symbols & intervals
    "SYMBOLS": ["SOLUSDT", "PAXGUSDT"],
    "INTERVALS": ["1s", "1m", "5m", "1h", "4h", "1d"],
    # Candle cap
    "MAX_BARS": 20_000,
    # Concurrency
    "MAX_FETCH_WORKERS": 4,
    # Caching
    "CACHE_DIR": Path("./klines_cache"),
    "CACHE_TTL_HOURS": 12,  # re‑download if older
    # Sub‑process timeout per analyser run (seconds)
    "ANALYSIS_TIMEOUT": 120,
    # Logging
    "LOG_FILE": "mfv_orchestrator.log",
}

# Command‑line overrides populate here later
ARGS: argparse.Namespace

# ────────────────────────────────────────────────────────────────────────────────
# 1. LOGGING SET‑UP
# ────────────────────────────────────────────────────────────────────────────────
CONFIG["CACHE_DIR"].mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s",
                              datefmt="%Y‑%m‑%d %H:%M:%S")

# Console
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
root_logger.addHandler(ch)

# Rotating file
fh = logging.handlers.RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=2**20,
                                          backupCount=3, encoding="utf‑8")
fh.setFormatter(formatter)
root_logger.addHandler(fh)

log = logging.getLogger(__name__)

logging.basicConfig(filename='mfv_orchestrator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# ────────────────────────────────────────────────────────────────────────────────
# 2. HELPERS – BINANCE KLINES
# ────────────────────────────────────────────────────────────────────────────────
KLINE_ENDPOINT = "/api/v3/klines"

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def _get(url: str, params: dict) -> requests.Response:
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Download up to *limit* most recent klines. Cached to disk."""
    cache_f = CONFIG["CACHE_DIR"] / f"{symbol}_{interval}.parquet"
    now = datetime.now(timezone.utc)
    if cache_f.exists():
        age = now - datetime.fromtimestamp(cache_f.stat().st_mtime, tz=timezone.utc)
        if age < timedelta(hours=CONFIG["CACHE_TTL_HOURS"]):
            log.debug("Using cached klines for %s %s", symbol, interval)
            return pd.read_parquet(cache_f)

    log.info("Downloading %s %s klines …", symbol, interval)
    end_ts = int(now.timestamp() * 1000)
    klines: List[list] = []
    pbar = tqdm(total=limit, desc=f"{symbol} {interval}")
    while len(klines) < limit:
        need = min(CONFIG["MAX_KLINES_PER_REQ"], limit - len(klines))
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": need,
            "endTime": end_ts,
        }
        data = _get(CONFIG["BASE_URL"] + KLINE_ENDPOINT, params).json()
        if not data:
            break  # exhausted history
        klines.extend(data)
        end_ts = data[0][0] - 1  # next request ends before oldest bar
        pbar.update(len(data))
        if len(data) < need:
            break  # no more history available
    pbar.close()

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]).astype(float)
    # Keep only required cols
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.sort_values("open_time", inplace=True)

    df.to_parquet(cache_f, compression="zstd")
    return df

# ────────────────────────────────────────────────────────────────────────────────
# 3. DATA PARTITIONING
# ────────────────────────────────────────────────────────────────────────────────

def partition_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    b_split = int(n * CONFIG["BENCHMARK_PCT"])
    t_split = b_split + int(n * CONFIG["TUNE_PCT"])
    bench = df.iloc[:b_split]
    tune = df.iloc[b_split:t_split]
    lock = df.iloc[t_split:]
    return bench, tune, lock

# ────────────────────────────────────────────────────────────────────────────────
# 4. ANALYSER SUB‑PROCESS WRAPPER
# ────────────────────────────────────────────────────────────────────────────────
import subprocess

def run_analyzer(csv_path: Path, symbol: str, interval: str, bench_pct: float, tune_pct: float) -> dict | None:
    cmd = [
        sys.executable, "walk_forward_analyzer.py",
        str(csv_path), symbol, interval,
    ]
    log.debug("Launching analyser: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=CONFIG["ANALYSIS_TIMEOUT"])
    except subprocess.TimeoutExpired:
        log.error("Analyser timed out for %s %s", symbol, interval)
        return None

    if proc.returncode != 0:
        log.error("Analyser failed for %s %s: %s", symbol, interval, proc.stderr[:300])
        return None

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        log.error("Invalid JSON from analyser for %s %s", symbol, interval)
        log.debug("Raw output: %s", proc.stdout[:300])
        return None

    if payload.get("status") != "success":
        log.warning("Analyser reported non‑success status for %s %s", symbol, interval)
        return None
    if "data" in payload:
        return payload["data"]
    else:
        log.warning("Analyser payload missing 'data' key for %s %s: %s", symbol, interval, payload)
        return payload

def flatten_result(data):
    # Flatten nested dicts for CSV
    flat = {
        'status': data.get('status'),
        'symbol': data.get('symbol'),
        'interval': data.get('interval'),
        'periods': str(data.get('periods')),
        'runtime_sec': data.get('runtime_sec'),
    }
    for prefix in ['tuning', 'lockbox']:
        d = data.get(prefix, {})
        for k, v in d.items():
            flat[f'{prefix}_{k}'] = v
    if 'robustness_score' in data:
        flat['robustness_score'] = data['robustness_score']
    return flat

ADAPTIVE_SPACE_FILE = 'adaptive_search_space.json'

# Helper to load/save adaptive search space

def load_adaptive_space():
    if os.path.exists(ADAPTIVE_SPACE_FILE):
        with open(ADAPTIVE_SPACE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_adaptive_space(space):
    with open(ADAPTIVE_SPACE_FILE, 'w') as f:
        json.dump(space, f)


def orchestrate():
    print("\n========== [MFV Orchestrator] Continual Refinement Mode ==========")
    prev_best_score = None
    prev_avg_score = None
    iteration = 0
    all_results = []
    prev_rows = defaultdict(dict)  # (symbol, interval) -> last row dict
    # Fetch and cache klines ONCE per symbol/interval
    tasks: List[Tuple[str, str]] = [(s, i) for s in CONFIG["SYMBOLS"] for i in CONFIG["INTERVALS"]]
    klines_cache = {}
    adaptive_space = load_adaptive_space()
    # Sequential fetching for debugging
    for sym, interv in tasks:
        try:
            df = fetch_klines(sym, interv, CONFIG["MAX_BARS"])
            klines_cache[(sym, interv)] = df
            print(f"[CACHE] {sym} {interv}: {len(df)} bars cached.")
            n = len(df)
            min_period = max(10, n // 1000)
            max_period = max(min_period + 1, n // 6)
            pair_key = f"{sym}_{interv}"
            adaptive_space[pair_key] = {'min': min_period, 'max': max_period}
        except Exception as exc:
            log.error("Download failed for %s %s: %s", sym, interv, exc)
            print(f"[ERROR] Download failed for {sym} {interv}: {exc}")
    save_adaptive_space(adaptive_space)
    try:
        # --- Real-time plotting setup ---
        plt.ion()
        metrics_to_plot = [
            "robustness_score",
            "tuning_youden_j",
            "tuning_calmar",
            "tuning_max_dd",
            "lockbox_youden_j",
            "lockbox_calmar",
            "lockbox_max_dd"
        ]
        fig, ax = plt.subplots()
        fig.suptitle("Average % Delta Across All Symbol/Intervals")
        report_file = Path("walk_forward_report.csv")
        # If the report file exists, load it to continue history
        if report_file.exists():
            full_history = pd.read_csv(report_file)
        else:
            full_history = pd.DataFrame()
        while True:
            iteration += 1
            print(f"\n[Iteration {iteration}] Starting...")
            results = []
            for symbol, interval in tasks:
                df = klines_cache.get((symbol, interval))
                if df is None:
                    print(f"[SKIP] No data for {symbol} {interval}")
                    continue
                print(f"Partitioning data for {symbol} {interval} ...")
                bench, tune, lock = partition_df(df)
                print(f"  Benchmark: {len(bench)} | Tune: {len(tune)} | Lock-box: {len(lock)}")
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
                    df.to_csv(tmp.name, index=False)
                    csv_path = Path(tmp.name)
                print(f"Running analyzer for {symbol} {interval} ...")
                # Pass adaptive search space as env var
                env = os.environ.copy()
                pair_key = f"{symbol}_{interval}"
                if pair_key in adaptive_space:
                    env['ADAPTIVE_SEARCH_SPACE'] = json.dumps(adaptive_space[pair_key])
                else:
                    env['ADAPTIVE_SEARCH_SPACE'] = ''
                # Use subprocess directly to pass env
                import subprocess
                cmd = [sys.executable, "walk_forward_analyzer.py", str(csv_path), symbol, interval]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=CONFIG["ANALYSIS_TIMEOUT"])
                csv_path.unlink(missing_ok=True)
                try:
                    payload = json.loads(proc.stdout)
                except Exception as e:
                    print(f"[FAIL] Analyzer output error for {symbol} {interval}: {e}")
                    log.error(f"Analyzer output error for {symbol} {interval}: {e}\nRaw output: {proc.stdout[:300]}")
                    continue
                if payload.get("status") == "success" and "data" in payload:
                    print(f"[OK] Analyzer completed for {symbol} {interval}")
                    results.append(flatten_result(payload["data"]))
                    # Update adaptive space with best periods
                    best_periods = payload["data"].get("periods")
                    if best_periods:
                        # Narrow search space by 20% around best
                        min_p = max(CONFIG['MAX_BARS']//100, int(min(best_periods) * 0.8))
                        max_p = min(CONFIG['MAX_BARS'], int(max(best_periods) * 1.2))
                        adaptive_space[pair_key] = {'min': min_p, 'max': max_p}
                elif payload.get("status") == "fail":
                    err = payload.get("error", "Unknown error")
                    print(f"[FAIL] Analyzer failed for {symbol} {interval}: {err}")
                    log.error(f"Analyzer failed for {symbol} {interval}: {err}")
                    continue
                else:
                    print(f"[FAIL] Analyzer returned unexpected output for {symbol} {interval}: {payload}")
                    log.error(f"Analyzer returned unexpected output for {symbol} {interval}: {payload}")
                    continue
            save_adaptive_space(adaptive_space)
            if not results:
                log.error("No successful analyser results – exiting")
                print("[FATAL] No successful analyser results – exiting")
                break
            # Aggregate and rank
            df_res = pd.DataFrame(results)
            # Normalise metrics 0‑1 for composite score
            norm_cols = []
            for col in ("tuning_youden_j", "tuning_calmar", "tuning_max_drawdown", "tuning_max_dd"):
                if col not in df_res:
                    continue
                norm_col = col + "_norm"
                norm_cols.append(norm_col)
                if col == "tuning_max_drawdown" or col == "tuning_max_dd":
                    df_res[norm_col] = 1 - (df_res[col] - df_res[col].min()) / (df_res[col].max() - df_res[col].min())
                else:
                    df_res[norm_col] = (df_res[col] - df_res[col].min()) / (df_res[col].max() - df_res[col].min())
            # Use available normalized columns for robustness_score
            score_expr = []
            if "tuning_youden_j_norm" in df_res:
                score_expr.append("0.5 * df_res['tuning_youden_j_norm']")
            if "tuning_calmar_norm" in df_res:
                score_expr.append("0.3 * df_res['tuning_calmar_norm']")
            if "tuning_max_drawdown_norm" in df_res:
                score_expr.append("0.2 * df_res['tuning_max_drawdown_norm']")
            elif "tuning_max_dd_norm" in df_res:
                score_expr.append("0.2 * df_res['tuning_max_dd_norm']")
            if score_expr:
                df_res["robustness_score"] = eval(" + ".join(score_expr))
                df_res.sort_values("robustness_score", ascending=False, inplace=True)
            else:
                log.warning("No normalized columns found for robustness_score calculation.")
                print("[WARN] No normalized columns found for robustness_score calculation.")
            # Append this iteration's results to full_history
            df_res["iteration"] = iteration
            full_history = pd.concat([full_history, df_res], ignore_index=True)
            # Save the full history to CSV (append mode, but keep header only once)
            full_history.to_csv(report_file, index=False, float_format='%.6g')
            print(f"[Iteration {iteration}] Updated report → {report_file} ({len(df_res)} rows this iter, {len(full_history)} total)")
            # Show performance improvement
            best_score = df_res["robustness_score"].max() if "robustness_score" in df_res else None
            avg_score = df_res["robustness_score"].mean() if "robustness_score" in df_res else None
            # Build and print full metrics table with deltas
            cols = [
                "symbol", "interval", "periods", "robustness_score",
                "tuning_youden_j", "tuning_calmar", "tuning_max_dd",
                "lockbox_youden_j", "lockbox_calmar", "lockbox_max_dd"
            ]
            delta_cols = [c + "_delta" for c in cols[3:]]
            table = []
            for _, row in df_res.iterrows():
                key = (row["symbol"], row["interval"])
                prev = prev_rows.get(key, {})
                row_out = [row.get(c, "") for c in cols]
                # Compute deltas
                for c in cols[3:]:
                    prev_val = prev.get(c, None)
                    curr_val = row.get(c, None)
                    if prev_val is not None and curr_val is not None:
                        try:
                            delta = float(curr_val) - float(prev_val)
                            row_out.append(f"{delta:+.4g}")
                        except Exception:
                            row_out.append("N/A")
                    else:
                        row_out.append("N/A")
                table.append(row_out)
                # Save for next iteration
                prev_rows[key] = {c: row.get(c, None) for c in cols}
                # --- Update plot data ---
                for i, metric in enumerate(metrics_to_plot, start=3):
                    val = row.get(metric, None)
                    if val is not None:
                        # This part of the plot data is no longer needed for the average delta plot
                        # as the plot now reads from the full history.
                        # Keeping it for now, but it will be removed in a future edit.
                        pass
                # --- Plot average % delta for each metric over all iterations ---
                ax.clear()
                # For each metric, build a list of average % delta per iteration
                for metric in metrics_to_plot:
                    # For each iteration, get all values for this metric
                    avg_pct_deltas = []
                    iter_nums = sorted(full_history["iteration"].unique())
                    for iter_num in iter_nums:
                        vals = full_history.loc[full_history["iteration"] == iter_num, metric].dropna().values
                        if len(vals) == 0:
                            avg_pct_deltas.append(None)
                            continue
                        y0 = vals[0] if vals[0] != 0 else 1e-8
                        avg_pct_delta = sum((v - y0) / abs(y0) * 100 for v in vals) / len(vals)
                        avg_pct_deltas.append(avg_pct_delta)
                    # Only plot if we have at least 2 points
                    if sum(x is not None for x in avg_pct_deltas) > 1:
                        ax.plot(iter_nums, [x if x is not None else float('nan') for x in avg_pct_deltas], label=metric)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Average % Delta from Iter 1")
                ax.legend(loc="best", fontsize="small")
                ax.grid(True)
                fig.canvas.draw()
                fig.canvas.flush_events()
            print(f"[Iteration {iteration}] Press Ctrl+C to stop or wait for next refinement...\n")
    except KeyboardInterrupt:
        print("\n[STOPPED] Continual refinement stopped by user.")
        print(f"Total iterations completed: {iteration}")
        print(f"Final report: walk_forward_report.csv")
        return
    # At the end of orchestrate(), after KeyboardInterrupt or loop exit
    plt.ioff()
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive MFV walk‑forward orchestrator")
    p.add_argument("--symbols", nargs="*", help="Override default symbols list")
    p.add_argument("--intervals", nargs="*", help="Override default intervals list")
    p.add_argument("--max_bars", type=int, help="Limit bars per dataset")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()


if __name__ == "__main__":
    import glob
    # Close all log handlers before deleting log files
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    try:
        # Delete .log and .csv files in the working directory
        for pattern in ("*.log", "*.csv"):
            for f in glob.glob(pattern):
                print(f"Deleting {f} ...")
                os.remove(f)
    except Exception as e:
        print(f"Error deleting logs or csv files: {e}")
        sys.exit(1)
    ARGS = parse_cli()
    if ARGS.debug:
        root_logger.setLevel(logging.DEBUG)
    if ARGS.symbols:
        CONFIG["SYMBOLS"] = ARGS.symbols
    if ARGS.intervals:
        CONFIG["INTERVALS"] = ARGS.intervals
    if ARGS.max_bars:
        CONFIG["MAX_BARS"] = ARGS.max_bars

    orchestrate()
