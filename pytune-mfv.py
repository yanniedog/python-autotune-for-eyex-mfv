import os
import sys
import argparse
import time
import datetime
import json
import logging
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Unified configuration settings for all data frequencies.
CONFIG = {
    # Analysis Parameters
    'FINAL_PEAK_COUNT': 8,
    'COMBINATION_SIZE': 4,
    'BROAD_SCAN_START': 10,
    'BROAD_SCAN_END': 5000,
    'BROAD_SCAN_STEP': 5,
    'MIN_PEAK_DISTANCE': 50,
    
    # Backtesting & Risk Management Parameters
    'STOP_LOSS_PCT': 5.0,
    'TAKE_PROFIT_PCT': 15.0,
    'IN_SAMPLE_PCT': 0.7,
    
    # Plotting Configuration
    'GENERATE_PLOT_FOR_SINGLE_MODE': True,
    
    # API & Data Parameters
    'BASE_URL': 'https://api.binance.com',
    'COMPREHENSIVE_SYMBOLS': ['SOLUSDT', 'PAXGUSDT'],
    'COMPREHENSIVE_INTERVALS': ['1s','1m','5m','1h','4h','1d'],
    'MAX_BARS_TO_DOWNLOAD': 10000
}

# ==============================================================================
# --- LOGGING SETUP ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mfv_analysis_robust.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# --- DATA FETCHING & VALIDATION ---
# ==============================================================================
session = requests.Session()
session.headers.update({'Accept-Encoding': 'gzip'})

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_request(url, params=None):
    """Makes a request with the shared session and handles retries."""
    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def get_exchange_info():
    """Fetches valid symbols from the exchange."""
    try:
        info = make_request(f"{CONFIG['BASE_URL']}/api/v3/exchangeInfo")
        valid_symbols = {s['symbol'] for s in info['symbols']}
        return valid_symbols
    except Exception as e:
        logging.error(f"Could not fetch exchange info: {e}. Proceeding without validation.")
        return None

def get_historical_klines(symbol, interval):
    """Fetches klines efficiently using the global configuration."""
    max_bars = CONFIG['MAX_BARS_TO_DOWNLOAD']
    logging.info(f"Fetching data for {symbol}, Interval: {interval} (max {max_bars} bars)...")
    klines = []
    limit = 1000
    end_time = int(time.time() * 1000)
    
    with tqdm(total=max_bars, desc=f"Downloading {symbol}/{interval}") as pbar:
        while len(klines) < max_bars:
            try:
                fetch_limit = min(limit, max_bars - len(klines))
                params = {'symbol': symbol, 'interval': interval, 'limit': fetch_limit, 'endTime': end_time}
                data = make_request(f"{CONFIG['BASE_URL']}/api/v3/klines", params=params)
                
                if not data:
                    logging.info("No more historical data available.")
                    break
                
                klines.extend(data)
                pbar.update(len(data))
                end_time = data[0][0] - 1
                
                if len(data) < fetch_limit:
                    break
            except Exception as e:
                logging.error(f"Error fetching klines for {symbol}/{interval}: {e}")
                return None
    
    sorted_klines = sorted(klines, key=lambda x: x[0])
    return sorted_klines[-max_bars:]

def process_klines_to_dataframe(klines):
    """Processes raw kline data into a memory-efficient DataFrame."""
    if not klines: return pd.DataFrame()
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbav', 'tbqv', 'ignore'])
    df = df[cols]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    df.dropna(inplace=True)
    start_date = pd.to_datetime(df['open_time'].iloc[0], unit='ms').strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df['close_time'].iloc[-1], unit='ms').strftime('%Y-%m-%d')
    logging.info(f"Processed {len(df)} klines from {start_date} to {end_date}.")
    return df

# ==============================================================================
# --- NUMBA ACCELERATED CORE BACKTESTING ENGINE ---
# ==============================================================================
@jit(nopython=True)
def _numba_backtest_loop(open_price, high_price, low_price, signal, horizon, sl_pct, tp_pct):
    """
    This function contains the core backtesting loop.
    The @jit decorator compiles it to high-speed machine code.
    """
    returns = np.zeros(len(open_price))
    trade_count = 0
    
    for i in range(1, len(open_price)):
        if signal[i-1] != 0:
            entry_price = open_price[i]
            trade_direction = np.sign(signal[i-1])
            outcome = 0.0

            if trade_direction > 0: # Long trade
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_price = entry_price * (1 + tp_pct / 100)
                
                for k in range(i, min(i + horizon, len(open_price))):
                    if low_price[k] <= sl_price:
                        outcome = -sl_pct
                        break
                    if high_price[k] >= tp_price:
                        outcome = tp_pct
                        break
                
                if outcome == 0.0:
                    exit_idx = min(i + horizon, len(open_price) - 1)
                    exit_price = open_price[exit_idx]
                    outcome = (exit_price / entry_price - 1) * 100
            
            returns[trade_count] = outcome
            trade_count += 1
            
    return returns[:trade_count]

def get_periods_per_year(interval_str):
    """Estimates the number of trading periods in a year for annualization."""
    if 'm' in interval_str: return 365 * 24 * (60 / int(interval_str.replace('m', '')))
    if 'h' in interval_str: return 365 * (24 / int(interval_str.replace('h', '')))
    if 'd' in interval_str: return 365
    if 's' in interval_str: return 365 * 24 * 60 * (60 / int(interval_str.replace('s', '')))
    return 252

def calculate_performance_metrics(df, signal_series, horizon, sl_pct, tp_pct, interval):
    """Prepares data for Numba engine and processes its results."""
    if df.empty or signal_series.empty: return {}, []

    open_price = df['open'].to_numpy()
    high_price = df['high'].to_numpy()
    low_price = df['low'].to_numpy()
    signal = signal_series.to_numpy()

    returns_np = _numba_backtest_loop(open_price, high_price, low_price, signal, horizon, sl_pct, tp_pct)
    
    if returns_np.size == 0: return {'youdens_j': -1, 'calmar_ratio': -999, 'max_drawdown_pct': 100}, []

    win_rate = np.sum(returns_np > 0) / len(returns_np) if returns_np.size > 0 else 0
    gross_profit = np.sum(returns_np[returns_np > 0]); gross_loss = np.abs(np.sum(returns_np[returns_np < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    equity_curve = np.cumsum(returns_np); running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve; max_drawdown_pct = np.max(drawdown) if drawdown.size > 0 else 0
    periods_per_year = get_periods_per_year(interval)
    num_years = len(signal_series) / periods_per_year if periods_per_year > 0 else 0
    total_return = equity_curve[-1] if equity_curve.size > 0 else 0
    annualized_return = (total_return / num_years) if num_years > 0 else 0
    calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else np.inf
    
    price_roc = df['close'].pct_change(periods=horizon).shift(-horizon) * 100
    combined = pd.DataFrame({'signal': signal_series, 'roc': price_roc}).dropna()
    signal_sign, roc_sign = np.sign(combined['signal']), np.sign(combined['roc'])
    tp = np.sum((signal_sign > 0) & (roc_sign > 0)); fn = np.sum((signal_sign <= 0) & (roc_sign > 0))
    tn = np.sum((signal_sign < 0) & (roc_sign < 0)); fp = np.sum((signal_sign >= 0) & (roc_sign < 0))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0; specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youdens_j = sensitivity + specificity - 1
    
    metrics = {'horizon': horizon, 'youdens_j': youdens_j, 'sensitivity': sensitivity, 'specificity': specificity, 'win_rate': win_rate, 'profit_factor': profit_factor, 'avg_return_pct': np.mean(returns_np), 'max_drawdown_pct': max_drawdown_pct, 'calmar_ratio': calmar_ratio, 'num_signals': len(returns_np)}
    return metrics, returns_np

# ==============================================================================
# --- ANALYSIS CORE (UNIFIED CONFIG) ---
# ==============================================================================
def run_vectorized_broad_scan(df):
    """Performs a broad scan using the global configuration."""
    start, end, step = CONFIG['BROAD_SCAN_START'], CONFIG['BROAD_SCAN_END'], CONFIG['BROAD_SCAN_STEP']
    logging.info(f"Starting vectorized broad scan from period {start} to {end} with step {step}...")
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan); mf_multiplier.fillna(0, inplace=True)
    mf_volume = mf_multiplier * volume
    results, scan_periods = [], np.arange(start, end + 1, step)
    for period in tqdm(scan_periods, desc="Vectorized Broad Scan"):
        if len(df) < period * 2: continue
        cum_mfv = mf_volume.rolling(window=period).sum()
        mean = cum_mfv.rolling(window=period).mean(); stdev = cum_mfv.rolling(window=period).std().replace(0, 1)
        mfv_line = ((cum_mfv - mean) / stdev).clip(-100, 100)
        price_roc = close.pct_change(periods=period) * 100
        combined = pd.DataFrame({'mfv': mfv_line, 'roc': price_roc}).dropna()
        if len(combined) < period: continue
        signal_sign, roc_sign = np.sign(combined['mfv']), np.sign(combined['roc'])
        tp = np.sum((signal_sign > 0) & (roc_sign > 0)); fn = np.sum((signal_sign <= 0) & (roc_sign > 0))
        tn = np.sum((signal_sign < 0) & (roc_sign < 0)); fp = np.sum((signal_sign >= 0) & (roc_sign < 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0; specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results.append({'period': period, 'youdens_j': sensitivity + specificity - 1})
    logging.info("Vectorized broad scan complete.")
    return results

def discover_best_horizon_for_combination(df, bar_ranges, interval):
    """Finds the best prediction horizon for a combination, optimizing for Youden's J."""
    if not bar_ranges: return None, []
    mf_volume = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)).fillna(0) * df['volume']
    def normalize_zscore(data, window):
        mean = data.rolling(window=window).mean(); stdev = data.rolling(window=window).std().replace(0, np.nan)
        return ((data - mean) / stdev).fillna(0)
    mfv_lines = [(normalize_zscore(mf_volume.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in bar_ranges]
    composite_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)
    min_horizon, max_horizon = min(bar_ranges), int(max(bar_ranges) * 1.2)
    horizon_test_range = range(min_horizon, max_horizon + 1, CONFIG['BROAD_SCAN_STEP'])
    best_result, best_returns, best_metric = None, [], -np.inf
    for horizon in horizon_test_range:
        if len(df) < horizon + max(bar_ranges): continue
        metrics, returns = calculate_performance_metrics(df, composite_signal, horizon, CONFIG['STOP_LOSS_PCT'], CONFIG['TAKE_PROFIT_PCT'], interval)
        if metrics and metrics.get('youdens_j', -np.inf) > best_metric:
            best_metric, best_result, best_returns = metrics['youdens_j'], metrics, returns
    return best_result, best_returns

def run_analysis(df, interval):
    """Core analysis logic to find best combinations using the global config."""
    scan_results = run_vectorized_broad_scan(df)
    if not scan_results: return []
    sorted_by_j = sorted(scan_results, key=lambda x: x['youdens_j'], reverse=True)
    diverse_peaks = []
    if sorted_by_j:
        diverse_peaks.append(sorted_by_j[0])
        for peak in sorted_by_j[1:]:
            if len(diverse_peaks) >= CONFIG['FINAL_PEAK_COUNT']: break
            if all(abs(peak['period'] - s_peak['period']) >= CONFIG['MIN_PEAK_DISTANCE'] for s_peak in diverse_peaks):
                diverse_peaks.append(peak)
    if len(diverse_peaks) < CONFIG['COMBINATION_SIZE']: return []
    peak_combinations = list(combinations([p['period'] for p in diverse_peaks], CONFIG['COMBINATION_SIZE']))
    logging.info(f"Analyzing {len(peak_combinations)} combinations...")
    all_combinations_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(discover_best_horizon_for_combination, df, combo, interval): combo for combo in peak_combinations}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing Combinations"):
            try:
                result, returns = future.result()
                if result: all_combinations_results.append({'combination': futures[future], 'metrics': result, 'returns': returns})
            except Exception as e: logging.error(f"Combination {futures[future]} failed: {e}")
    return sorted(all_combinations_results, key=lambda x: x['metrics'].get('youdens_j', -np.inf), reverse=True)

# ==============================================================================
# --- PLOTTING & REPORTING ---
# ==============================================================================
def make_json_serializable(obj):
    """Recursively converts numpy types to native Python types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    # FIX: Explicitly handle tuples to convert their contents
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return obj
    
def generate_equity_curve_plot(top_candidate, symbol, interval):
    """Generates and saves a plot of the equity curve and drawdown."""
    returns = top_candidate['returns']
    if len(returns) == 0: logging.warning("Cannot generate plot: No returns data."); return
    metrics, combo_str = top_candidate['metrics'], ", ".join(map(str, sorted(top_candidate['combination'])))
    equity_curve, running_max = np.cumsum(returns), np.maximum.accumulate(np.cumsum(returns))
    drawdown = running_max - equity_curve
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(equity_curve, label='Equity Curve', color='cyan')
    ax1.set_title(f'Equity Curve & Drawdown for {symbol}/{interval}\nCombination: {combo_str} | Horizon: {metrics["horizon"]}', fontsize=16)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12); ax1.legend()
    ax2.fill_between(range(len(drawdown)), -drawdown, 0, color='red', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)', fontsize=12); ax2.set_xlabel('Trade Number', fontsize=12)
    plt.tight_layout(); filename = f'equity_curve_{symbol}_{interval}.png'
    plt.savefig(filename); logging.info(f"Equity curve plot saved to '{filename}'"); plt.close()

def display_single_run_report(results, symbol, interval):
    """Displays a report for a single analysis run, ranked by Youden's J."""
    if not results: logging.info("No profitable combinations found."); return
    print("\n" + "="*160 + f"\n--- TOP 10 PEAK COMBINATIONS FOR {symbol}/{interval} (RANKED BY YOUDEN'S J) ---".center(160) + "\n" + "="*160)
    header = f"{'Rank':<5} | {'Youden\'s J':<11} | {'Calmar Ratio':<14} | {'Max DD %':<10} | {'Horizon':<8} | {'Win Rate':<10} | {'Profit F.':<11} | {'Num Signals':<12} | {'MFV Combination'}"
    print(header + "\n" + "-" * len(header))
    for i, result in enumerate(results[:10], 1):
        m = result['metrics']
        youdens_j = f"{m['youdens_j']:.3f}"
        calmar = f"{m['calmar_ratio']:.2f}" if np.isfinite(m['calmar_ratio']) else "inf"
        max_dd = f"{m['max_drawdown_pct']:.2f}%"
        horizon = f"{m['horizon']} bars"
        win_rate = f"{m['win_rate']:.1%}"
        profit_factor = f"{m['profit_factor']:.2f}" if np.isfinite(m['profit_factor']) else "inf"
        num_signals = f"{m['num_signals']}"
        combo_str = ", ".join(map(str, sorted(result['combination'])))
        print(f"{i:<5} | {youdens_j:<11} | {calmar:<14} | {max_dd:<10} | {horizon:<8} | {win_rate:<10} | {profit_factor:<11} | {num_signals:<12} | {combo_str}")
    print("="*160)

def display_comprehensive_report(all_results_df):
    """Displays the final meta-analysis report, with a robustness score weighted towards Youden's J."""
    if all_results_df.empty: logging.warning("No data for meta-analysis."); return
    df = all_results_df.copy()
    
    # Normalize metrics for fair scoring
    df['youdens_j_norm'] = (df['youdens_j'] - df['youdens_j'].min()) / (df['youdens_j'].max() - df['youdens_j'].min())
    df['oos_calmar_ratio_norm'] = (df['oos_calmar_ratio'] - df['oos_calmar_ratio'].min()) / (df['oos_calmar_ratio'].max() - df['oos_calmar_ratio'].min())
    df['oos_max_drawdown_pct_inv_norm'] = 1 - ((df['oos_max_drawdown_pct'] - df['oos_max_drawdown_pct'].min()) / (df['oos_max_drawdown_pct'].max() - df['oos_max_drawdown_pct'].min()))
    
    # Calculate robustness score with 50% weight on Youden's J
    df['robustness_score'] = 0.5 * df['youdens_j_norm'].fillna(0) + 0.3 * df['oos_calmar_ratio_norm'].fillna(0) + 0.2 * df['oos_max_drawdown_pct_inv_norm'].fillna(0)
    df = df.sort_values('robustness_score', ascending=False)
    
    print("\n" + "#"*170 + "\n" + "--- COMPREHENSIVE META-ANALYSIS REPORT (SCORE: 50% Youden's J, 30% OOS Calmar, 20% OOS Drawdown) ---".center(170) + "\n" + "#"*170)
    header = f"{'Rank':<5} | {'Symbol':<10} | {'Interval':<8} | {'Robust Score':<14} | {'Youden\'s J (IS)':<16} | {'Calmar (OOS)':<14} | {'Max DD % (OOS)':<16} | {'Win Rate (OOS)':<16} | {'MFV Combination'}"
    print(header + "\n" + "-" * len(header))
    for i, row in enumerate(df.head(20).itertuples(), 1):
        robust_score = f"{row.robustness_score:.3f}"
        youdens_j_is = f"{row.youdens_j:.3f}"
        calmar_oos = f"{row.oos_calmar_ratio:.2f}" if np.isfinite(row.oos_calmar_ratio) else "inf"
        max_dd_oos = f"{row.oos_max_drawdown_pct:.2f}%"
        win_rate_oos = f"{row.oos_win_rate:.1%}"
        print(f"{i:<5} | {row.symbol:<10} | {row.interval:<8} | {robust_score:<14} | {youdens_j_is:<16} | {calmar_oos:<14} | {max_dd_oos:<16} | {win_rate_oos:<16} | {row.combination}")
    print("#"*170)

def save_comprehensive_results(results):
    """Saves comprehensive results to JSON and CSV."""
    logging.info("Saving comprehensive report data...")
    with open('comprehensive_report_robust.json', 'w') as f: json.dump(make_json_serializable(results), f, indent=4)
    flat_combinations = []
    for run in results:
        for combo_result in run.get('robust_combinations', []):
             flat_combinations.append({'symbol': run['symbol'], 'interval': run['interval'], 'combination': ", ".join(map(str, sorted(combo_result['combination']))), **combo_result['in_sample_metrics'], **{'oos_' + k: v for k, v in combo_result['out_of_sample_metrics'].items()}})
    if flat_combinations: pd.DataFrame(flat_combinations).to_csv('comprehensive_report_robust.csv', index=False)
    logging.info("Data saved to 'comprehensive_report_robust.json' and 'comprehensive_report_robust.csv'")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
def main():
    """Main function to parse arguments and run the selected mode."""
    parser = argparse.ArgumentParser(description="Robust, High-Performance MFV Analysis Engine with Risk Management.")
    parser.add_argument('mode', nargs='?', default=None, choices=['single', 'comprehensive'], help="Choose analysis mode.")
    parser.add_argument('--symbol', type=str, help="Symbol for 'single' mode (e.g., BTCUSDT).")
    parser.add_argument('--interval', type=str, help="Interval for 'single' mode (e.g., 1h).")
    args = parser.parse_args()

    if args.mode is None:
        while True:
            print("\nSelect Analysis Mode:\n1. Single Analysis (Interactive, with plotting)\n2. Comprehensive Report (Automated batch analysis with Out-of-Sample validation)")
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1': args.mode = 'single'; break
            elif choice == '2': args.mode = 'comprehensive'; break
            else: print("Invalid choice.")

    valid_symbols = get_exchange_info()

    if args.mode == 'single':
        if not args.symbol: args.symbol = input("Enter symbol (e.g., BTCUSDT): ").upper()
        if not args.interval: args.interval = input("Enter interval (e.g., 1h, 4h, 1d): ").lower()
        if valid_symbols and args.symbol not in valid_symbols: logging.error(f"Invalid symbol: {args.symbol}."); return
        df = process_klines_to_dataframe(get_historical_klines(args.symbol, args.interval))
        if df.empty: return
        results = run_analysis(df, args.interval)
        display_single_run_report(results, args.symbol, args.interval)
        if results and CONFIG['GENERATE_PLOT_FOR_SINGLE_MODE']:
            generate_equity_curve_plot(results[0], args.symbol, args.interval)

    elif args.mode == 'comprehensive':
        all_run_results = []
        for symbol in CONFIG['COMPREHENSIVE_SYMBOLS']:
            if valid_symbols and symbol.upper() not in valid_symbols: logging.warning(f"Invalid symbol: {symbol}. Skipping."); continue
            for interval in CONFIG['COMPREHENSIVE_INTERVALS']:
                logging.info(f"--- STARTING: {symbol} | {interval} ---")
                df_full = process_klines_to_dataframe(get_historical_klines(symbol, interval))
                if df_full.empty or len(df_full) < 200: logging.warning(f"Skipping {symbol}/{interval} due to insufficient data."); continue
                split_idx = int(len(df_full) * CONFIG['IN_SAMPLE_PCT'])
                df_is, df_oos = df_full.iloc[:split_idx].copy(), df_full.iloc[split_idx:].copy()
                logging.info(f"Data split: {len(df_is)} IS bars, {len(df_oos)} OOS bars.")
                is_results = run_analysis(df_is, interval)
                if not is_results: logging.warning(f"No promising combinations in-sample for {symbol}/{interval}."); continue
                logging.info(f"Testing top {min(10, len(is_results))} candidates out-of-sample...")
                robust_combinations = []
                for candidate in tqdm(is_results[:10], desc="Out-of-Sample Validation"):
                    mf_volume_oos = (((df_oos['close'] - df_oos['low']) - (df_oos['high'] - df_oos['close'])) / (df_oos['high'] - df_oos['low']).replace(0, np.nan)).fillna(0) * df_oos['volume']
                    def normalize_zscore(data, window):
                        mean = data.rolling(window=window).mean(); stdev = data.rolling(window=window).std().replace(0, np.nan)
                        return ((data - mean) / stdev).fillna(0)
                    mfv_lines = [(normalize_zscore(mf_volume_oos.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in candidate['combination']]
                    oos_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)
                    oos_metrics, _ = calculate_performance_metrics(df_oos, oos_signal, candidate['metrics']['horizon'], CONFIG['STOP_LOSS_PCT'], CONFIG['TAKE_PROFIT_PCT'], interval)
                    if oos_metrics: robust_combinations.append({'combination': candidate['combination'], 'in_sample_metrics': candidate['metrics'], 'out_of_sample_metrics': oos_metrics})
                all_run_results.append({"symbol": symbol, "interval": interval, "robust_combinations": robust_combinations})
                save_comprehensive_results(all_run_results)
        logging.info("All analyses complete. Generating final meta-analysis report...")
        flat_results = []
        for run in all_run_results:
            for combo in run.get('robust_combinations', []):
                flat_results.append({'symbol': run['symbol'], 'interval': run['interval'], 'combination': ", ".join(map(str, sorted(combo['combination']))), **combo['in_sample_metrics'], **{'oos_' + k: v for k, v in combo['out_of_sample_metrics'].items()}})
        if flat_results: display_comprehensive_report(pd.DataFrame(flat_results))
        else: logging.warning("No robust combinations found across all runs.")

if __name__ == '__main__':
    freeze_support()
    main()
