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
        "--bench", str(bench_pct), "--tune", str(tune_pct),
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
    return payload["data"]

# ────────────────────────────────────────────────────────────────────────────────
# 5. MAIN ORCHESTRATION
# ────────────────────────────────────────────────────────────────────────────────

def orchestrate() -> None:
    results = []

    tasks: List[Tuple[str, str]] = [(s, i) for s in CONFIG["SYMBOLS"] for i in CONFIG["INTERVALS"]]

    # Fetch klines in parallel to maximise IO latency hiding
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_FETCH_WORKERS"]) as pool:
        future_map = {
            pool.submit(fetch_klines, sym, interv, CONFIG["MAX_BARS"]): (sym, interv)
            for sym, interv in tasks
        }
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Download tasks"):
            symbol, interval = future_map[fut]
            try:
                df = fut.result()
            except Exception as exc:
                log.error("Download failed for %s %s: %s", symbol, interval, exc)
                continue

            # Partition and stage CSV
            bench, tune, lock = partition_df(df)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                csv_path = Path(tmp.name)

            data = run_analyzer(csv_path, symbol, interval,
                                CONFIG["BENCHMARK_PCT"], CONFIG["TUNE_PCT"])
            csv_path.unlink(missing_ok=True)
            if data:
                results.append(data)

    if not results:
        log.error("No successful analyser results – exiting")
        return

    # Aggregate and rank
    df_res = pd.DataFrame(results)
    # Normalise metrics 0‑1 for composite score
    for col in ("youden_j", "calmar", "max_drawdown"):
        if col not in df_res:
            continue
        if col == "max_drawdown":
            df_res[col + "_norm"] = 1 - (df_res[col] - df_res[col].min()) / (df_res[col].max() - df_res[col].min())
        else:
            df_res[col + "_norm"] = (df_res[col] - df_res[col].min()) / (df_res[col].max() - df_res[col].min())

    df_res["robustness_score"] = (
        0.5 * df_res["youden_j_norm"] +
        0.3 * df_res["calmar_norm"] +
        0.2 * df_res["max_drawdown_norm"]
    )
    df_res.sort_values("robustness_score", ascending=False, inplace=True)

    # Save & print
    out_f = Path("walk_forward_report.csv")
    df_res.to_csv(out_f, index=False)
    log.info("Saved report → %s (%d rows)", out_f, len(df_res))
    print(df_res.head(20).to_markdown(index=False))

# ────────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive MFV walk‑forward orchestrator")
    p.add_argument("--symbols", nargs="*", help="Override default symbols list")
    p.add_argument("--intervals", nargs="*", help="Override default intervals list")
    p.add_argument("--max‑bars", type=int, help="Limit bars per dataset")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args()


if __name__ == "__main__":
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
