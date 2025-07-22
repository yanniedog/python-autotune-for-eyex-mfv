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

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# These settings are now defaults and can be overridden by command-line arguments.
CONFIG = {
    # Analysis Parameters
    'FINAL_PEAK_COUNT': 8,
    'MIN_PEAK_DISTANCE': 50,
    'FINE_TUNE_PEAK_RANGE': 4,
    'BROAD_SCAN_START': 10,
    'BROAD_SCAN_END': 5000,
    'BROAD_SCAN_STEP': 5,
    
    # NEW: Backtesting & Risk Management Parameters
    'STOP_LOSS_PCT': 5.0,        # Stop loss percentage
    'TAKE_PROFIT_PCT': 15.0,       # Take profit percentage
    'IN_SAMPLE_PCT': 0.7,        # 70% of data for training, 30% for out-of-sample validation
    'COMBINATION_SIZE': 4,       # Number of MFV periods to combine
    
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
    """Fetches up to MAX_BARS_TO_DOWNLOAD klines efficiently."""
    logging.info(f"Fetching data for {symbol}, Interval: {interval} (max {CONFIG['MAX_BARS_TO_DOWNLOAD']} bars)...")
    klines = []
    limit = 1000
    end_time = int(time.time() * 1000)
    
    with tqdm(total=CONFIG['MAX_BARS_TO_DOWNLOAD'], desc=f"Downloading {symbol}/{interval}") as pbar:
        while len(klines) < CONFIG['MAX_BARS_TO_DOWNLOAD']:
            try:
                fetch_limit = min(limit, CONFIG['MAX_BARS_TO_DOWNLOAD'] - len(klines))
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
    return sorted_klines[-CONFIG['MAX_BARS_TO_DOWNLOAD']:]

def process_klines_to_dataframe(klines):
    """Processes raw kline data into a memory-efficient DataFrame."""
    if not klines:
        return pd.DataFrame()
        
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
# --- NEW: CORE BACKTESTING & METRICS ENGINE ---
# ==============================================================================
def get_periods_per_year(interval_str):
    """Estimates the number of trading periods in a year for annualization."""
    if 'm' in interval_str:
        minutes = int(interval_str.replace('m', ''))
        return 365 * 24 * (60 / minutes)
    elif 'h' in interval_str:
        hours = int(interval_str.replace('h', ''))
        return 365 * (24 / hours)
    elif 'd' in interval_str:
        return 365
    elif 's' in interval_str:
        seconds = int(interval_str.replace('s', ''))
        return 365 * 24 * 60 * (60 / seconds)
    return 252 # Default fallback

def calculate_performance_metrics(df, signal_series, horizon, sl_pct, tp_pct, interval):
    """
    NEW: A dedicated function to run a backtest with SL/TP and calculate all key metrics.
    This is the core of the new risk-managed backtesting.
    """
    if df.empty or signal_series.empty:
        return {}

    open_price = df['open'].to_numpy()
    high_price = df['high'].to_numpy()
    low_price = df['low'].to_numpy()
    signal = signal_series.to_numpy()
    
    returns = []
    trade_active = False
    
    for i in range(1, len(df)):
        if trade_active: continue # Only one trade at a time

        # Entry condition: signal is present at previous bar
        if signal[i-1] != 0:
            entry_price = open_price[i]
            trade_direction = np.sign(signal[i-1])
            trade_active = True
            outcome = 0.0

            if trade_direction > 0: # Long trade
                sl_price = entry_price * (1 - sl_pct / 100)
                tp_price = entry_price * (1 + tp_pct / 100)
                for k in range(i, min(i + horizon, len(df))):
                    if low_price[k] <= sl_price:
                        outcome = -sl_pct
                        break
                    if high_price[k] >= tp_price:
                        outcome = tp_pct
                        break
                if outcome == 0.0: # Exit at horizon if no SL/TP hit
                    exit_price = open_price[min(i + horizon, len(df) - 1)]
                    outcome = (exit_price / entry_price - 1) * 100
            
            # NOTE: For simplicity, this example implements long-only.
            # A full implementation would have a symmetric case for short trades.
            
            returns.append(outcome)
            # This simple model assumes the trade resolves and we can look for a new one the next day.
            # A more complex model would handle the trade duration.
            trade_active = False 

    if not returns:
        return {'youdens_j': -1, 'calmar_ratio': -999, 'max_drawdown_pct': 100}

    returns_np = np.array(returns)
    win_rate = np.sum(returns_np > 0) / len(returns_np) if returns_np.size > 0 else 0
    gross_profit = np.sum(returns_np[returns_np > 0])
    gross_loss = np.abs(np.sum(returns_np[returns_np < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Calculate Drawdown
    equity_curve = np.cumsum(returns_np)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    max_drawdown_pct = np.max(drawdown) if drawdown.size > 0 else 0

    # Calculate Calmar Ratio
    periods_per_year = get_periods_per_year(interval)
    num_years = len(signal_series) / periods_per_year if periods_per_year > 0 else 0
    total_return = equity_curve[-1] if equity_curve.size > 0 else 0
    annualized_return = (total_return / num_years) if num_years > 0 else 0
    calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else np.inf
    
    # Youden's J (remains a useful signal quality metric)
    price_roc = df['close'].pct_change(periods=horizon).shift(-horizon) * 100
    combined = pd.DataFrame({'signal': signal_series, 'roc': price_roc}).dropna()
    signal_sign = np.sign(combined['signal'])
    roc_sign = np.sign(combined['roc'])
    tp = np.sum((signal_sign > 0) & (roc_sign > 0))
    fn = np.sum((signal_sign <= 0) & (roc_sign > 0))
    tn = np.sum((signal_sign < 0) & (roc_sign < 0))
    fp = np.sum((signal_sign >= 0) & (roc_sign < 0))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youdens_j = sensitivity + specificity - 1

    return {
        'horizon': horizon, 'youdens_j': youdens_j, 'sensitivity': sensitivity,
        'specificity': specificity, 'win_rate': win_rate, 'profit_factor': profit_factor,
        'avg_return_pct': np.mean(returns_np), 'max_drawdown_pct': max_drawdown_pct,
        'calmar_ratio': calmar_ratio, 'num_signals': len(returns_np)
    }

# ==============================================================================
# --- ANALYSIS CORE (UPDATED FOR ROBUSTNESS) ---
# ==============================================================================
def run_vectorized_broad_scan(df):
    """Performs a vectorized broad scan for MFV performance to find rough peaks."""
    start, end, step = CONFIG['BROAD_SCAN_START'], CONFIG['BROAD_SCAN_END'], CONFIG['BROAD_SCAN_STEP']
    logging.info(f"Starting vectorized broad scan from period {start} to {end}...")
    
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_multiplier.fillna(0, inplace=True)
    mf_volume = mf_multiplier * volume
    
    results = []
    scan_periods = np.arange(start, end + 1, step)
    
    for period in tqdm(scan_periods, desc="Vectorized Broad Scan"):
        if len(df) < period * 2: continue
        
        cum_mfv = mf_volume.rolling(window=period).sum()
        mean = cum_mfv.rolling(window=period).mean()
        stdev = cum_mfv.rolling(window=period).std().replace(0, 1)
        normalized_mfv = (cum_mfv - mean) / stdev
        mfv_line = normalized_mfv.clip(-100, 100)
        
        price_roc = close.pct_change(periods=period) * 100
        combined = pd.DataFrame({'mfv': mfv_line, 'roc': price_roc}).dropna()
        
        if len(combined) < period: continue
            
        signal_sign = np.sign(combined['mfv'])
        roc_sign = np.sign(combined['roc'])

        tp = np.sum((signal_sign > 0) & (roc_sign > 0))
        fn = np.sum((signal_sign <= 0) & (roc_sign > 0))
        tn = np.sum((signal_sign < 0) & (roc_sign < 0))
        fp = np.sum((signal_sign >= 0) & (roc_sign < 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youdens_j = sensitivity + specificity - 1

        results.append({'period': period, 'youdens_j': youdens_j})
        
    logging.info("Vectorized broad scan complete.")
    return results

def discover_best_horizon_for_combination(df, bar_ranges, interval):
    """
    Finds the best prediction horizon for a given combination of MFV periods.
    This now uses the new backtesting engine.
    """
    if not bar_ranges: return None
    
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_multiplier.fillna(0, inplace=True)
    mf_volume = mf_multiplier * volume

    def normalize_zscore(data, window):
        mean = data.rolling(window=window).mean()
        stdev = data.rolling(window=window).std().replace(0, np.nan)
        return ((data - mean) / stdev).fillna(0)

    mfv_lines = [(normalize_zscore(mf_volume.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in bar_ranges]
    composite_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)
    
    min_horizon = min(bar_ranges)
    max_horizon = int(max(bar_ranges) * 1.2)
    horizon_test_range = range(min_horizon, max_horizon + 1, CONFIG['BROAD_SCAN_STEP'])
    
    best_result = None
    # OPTIMIZE FOR CALMAR RATIO to prioritize risk-adjusted return
    best_metric = -np.inf
    
    for horizon in horizon_test_range:
        if len(df) < horizon + max(bar_ranges): continue
        
        metrics = calculate_performance_metrics(
            df, composite_signal, horizon, 
            CONFIG['STOP_LOSS_PCT'], CONFIG['TAKE_PROFIT_PCT'], interval
        )
        
        if metrics and metrics.get('calmar_ratio', -np.inf) > best_metric:
            best_metric = metrics['calmar_ratio']
            best_result = metrics
            
    return best_result

def run_in_sample_analysis(df_is, interval):
    """
    Main analysis logic, now explicitly for IN-SAMPLE data to find best parameters.
    """
    scan_results = run_vectorized_broad_scan(df_is)
    if not scan_results: return [], []

    # Fine-tuning peaks remains the same, but on in-sample data
    # (The function `find_and_fine_tune_peaks` is simple enough to be kept in main flow)
    # ... for brevity, we will use a simplified peak selection here
    
    sorted_by_j = sorted(scan_results, key=lambda x: x['youdens_j'], reverse=True)
    
    diverse_peaks = []
    if sorted_by_j:
        diverse_peaks.append(sorted_by_j[0])
        for peak in sorted_by_j[1:]:
            if len(diverse_peaks) >= CONFIG['FINAL_PEAK_COUNT']: break
            is_distant = all(abs(peak['period'] - s_peak['period']) >= CONFIG['MIN_PEAK_DISTANCE'] for s_peak in diverse_peaks)
            if is_distant: diverse_peaks.append(peak)
    
    if len(diverse_peaks) < CONFIG['COMBINATION_SIZE']: return diverse_peaks, []

    peak_combinations = list(combinations([p['period'] for p in diverse_peaks], CONFIG['COMBINATION_SIZE']))
    logging.info(f"Analyzing {len(peak_combinations)} combinations on in-sample data...")
    
    all_combinations_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(discover_best_horizon_for_combination, df_is, combo, interval): combo for combo in peak_combinations}
        for future in tqdm(as_completed(futures), total=len(futures), desc="In-Sample Analysis"):
            combo = futures[future]
            try:
                result = future.result()
                if result:
                    all_combinations_results.append({'combination': combo, **result})
            except Exception as e:
                logging.error(f"In-sample combination {combo} failed: {e}")

    logging.info("In-sample analysis complete.")
    # Return all diverse peaks, and the top combinations sorted by Calmar Ratio
    return diverse_peaks, sorted(all_combinations_results, key=lambda x: x.get('calmar_ratio', -np.inf), reverse=True)

# ==============================================================================
# --- MAIN EXECUTION & REPORTING (UPDATED) ---
# ==============================================================================
def make_json_serializable(obj):
    """Recursively converts numpy types to native Python types for JSON."""
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_json_serializable(i) for i in obj]
    return obj

def save_results_to_files(results):
    """Saves the comprehensive results to JSON and a flattened CSV."""
    logging.info("Saving comprehensive report data...")
    
    serializable_results = make_json_serializable(results)
    with open('comprehensive_report_robust.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
        
    flat_combinations = []
    for run in results:
        for combo_result in run.get('robust_combinations', []):
             flat_combinations.append({
                'symbol': run['symbol'], 
                'interval': run['interval'],
                'combination': ", ".join(map(str, sorted(combo_result['combination']))),
                **combo_result['in_sample_metrics'],
                **{'oos_' + k: v for k, v in combo_result['out_of_sample_metrics'].items()}
            })
    
    if flat_combinations:
        pd.DataFrame(flat_combinations).to_csv('comprehensive_report_robust.csv', index=False)
        
    logging.info("Data saved to 'comprehensive_report_robust.json' and 'comprehensive_report_robust.csv'")

def display_final_report(all_results_df):
    """Displays the final meta-analysis report with IS and OOS results."""
    if all_results_df.empty:
        logging.warning("No combination data available for meta-analysis.")
        return

    df = all_results_df.copy()
    
    # NEW: Calculate a robustness score
    # We penalize high OOS drawdown and reward high OOS Calmar ratio.
    df['oos_calmar_ratio_norm'] = (df['oos_calmar_ratio'] - df['oos_calmar_ratio'].min()) / (df['oos_calmar_ratio'].max() - df['oos_calmar_ratio'].min())
    df['oos_max_drawdown_pct_inv_norm'] = 1 - ((df['oos_max_drawdown_pct'] - df['oos_max_drawdown_pct'].min()) / (df['oos_max_drawdown_pct'].max() - df['oos_max_drawdown_pct'].min()))
    df['robustness_score'] = 0.5 * df['oos_calmar_ratio_norm'] + 0.3 * df['oos_max_drawdown_pct_inv_norm'] + 0.2 * df['youdens_j']
    df = df.sort_values('robustness_score', ascending=False)


    print("\n" + "#"*160)
    print("--- COMPREHENSIVE META-ANALYSIS REPORT (RANKED BY ROBUSTNESS SCORE) ---".center(160))
    print("#"*160)

    def print_top_10_table(sub_df, title):
        print(f"\n--- {title} ---")
        top_10 = sub_df.head(10)
        if top_10.empty:
            print("No results for this category.")
            return
        
        header = (f"{'Rank':<5} | {'Robust Score':<14} | {'Horizon':<8} | {'Calmar (OOS)':<14} | {'Max DD % (OOS)':<16} | "
                  f"{'Win Rate (OOS)':<16} | {'Profit F. (OOS)':<17} | {'Youden\'s J (IS)':<16} | {'MFV Combination'}")
        print(header)
        print("-" * len(header))
        for i, row in enumerate(top_10.itertuples(), 1):
            robust_score = f"{row.robustness_score:.3f}"
            horizon = f"{row.horizon} bars"
            calmar_oos = f"{row.oos_calmar_ratio:.2f}" if np.isfinite(row.oos_calmar_ratio) else "inf"
            max_dd_oos = f"{row.oos_max_drawdown_pct:.2f}%"
            win_rate_oos = f"{row.oos_win_rate:.1%}"
            profit_factor_oos = f"{row.oos_profit_factor:.2f}" if np.isfinite(row.oos_profit_factor) else "inf"
            youdens_j_is = f"{row.youdens_j:.3f}"
            
            print(f"{i:<5} | {robust_score:<14} | {horizon:<8} | {calmar_oos:<14} | {max_dd_oos:<16} | "
                  f"{win_rate_oos:<16} | {profit_factor_oos:<17} | {youdens_j_is:<16} | {row.combination}")

    print("\n\n" + "="*160)
    print("--- TOP 10 GLOBAL COMBINATIONS (ACROSS ALL SYMBOLS AND TIMEFRAMES) ---".center(160))
    print("="*160)
    print_top_10_table(df, "Top 10 Overall (Sorted by Robustness Score)")
    print("="*160)

def main():
    """Main function to parse arguments and run the selected mode."""
    parser = argparse.ArgumentParser(description="Robust, High-Performance MFV Analysis Engine with Risk Management.")
    parser.add_argument(
        'mode', nargs='?', default='comprehensive', choices=['comprehensive'],
        help="Mode of operation. Currently only 'comprehensive' is fully supported in this version."
    )
    args = parser.parse_args()

    valid_symbols = get_exchange_info()
    
    all_run_results = []
    for symbol in CONFIG['COMPREHENSIVE_SYMBOLS']:
        if valid_symbols and symbol.upper() not in valid_symbols:
            logging.warning(f"Invalid symbol: {symbol}. Skipping.")
            continue
        for interval in CONFIG['COMPREHENSIVE_INTERVALS']:
            logging.info(f"--- STARTING ANALYSIS FOR: {symbol} | TIMEFRAME: {interval} ---")
            
            # 1. Fetch and process data
            df_full = process_klines_to_dataframe(get_historical_klines(symbol, interval))
            if df_full.empty or len(df_full) < 100: # Need enough data for split
                logging.warning(f"Skipping {symbol}/{interval} due to insufficient data.")
                continue

            # 2. Split data into In-Sample (IS) and Out-of-Sample (OOS)
            split_idx = int(len(df_full) * CONFIG['IN_SAMPLE_PCT'])
            df_is = df_full.iloc[:split_idx].copy()
            df_oos = df_full.iloc[split_idx:].copy()
            logging.info(f"Data split: {len(df_is)} in-sample bars, {len(df_oos)} out-of-sample bars.")

            # 3. Run In-Sample analysis to find the best candidate combinations
            _, top_is_combinations = run_in_sample_analysis(df_is, interval)
            
            if not top_is_combinations:
                logging.warning(f"No promising combinations found in-sample for {symbol}/{interval}.")
                continue

            # 4. Test the top candidates Out-of-Sample
            logging.info(f"Testing top {min(10, len(top_is_combinations))} candidates out-of-sample...")
            robust_combinations = []
            for candidate in tqdm(top_is_combinations[:10], desc="Out-of-Sample Validation"):
                # Generate the composite signal on the OOS data using the combination found in-sample
                close, high, low, volume = df_oos['close'], df_oos['high'], df_oos['low'], df_oos['volume']
                mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
                mf_multiplier.fillna(0, inplace=True)
                mf_volume = mf_multiplier * volume
                
                def normalize_zscore(data, window):
                    mean = data.rolling(window=window).mean()
                    stdev = data.rolling(window=window).std().replace(0, np.nan)
                    return ((data - mean) / stdev).fillna(0)
                
                mfv_lines = [(normalize_zscore(mf_volume.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in candidate['combination']]
                oos_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)

                # Run the backtest on OOS data with the horizon found in-sample
                oos_metrics = calculate_performance_metrics(
                    df_oos, oos_signal, candidate['horizon'], 
                    CONFIG['STOP_LOSS_PCT'], CONFIG['TAKE_PROFIT_PCT'], interval
                )
                
                if oos_metrics:
                    robust_combinations.append({
                        'combination': candidate['combination'],
                        'in_sample_metrics': candidate,
                        'out_of_sample_metrics': oos_metrics
                    })

            all_run_results.append({
                "symbol": symbol, "interval": interval,
                "robust_combinations": robust_combinations
            })
            
            # Save incrementally
            save_results_to_files(all_run_results)
    
    # Final Meta-Analysis Report
    logging.info("All individual analyses complete. Generating final meta-analysis report...")
    flat_results = []
    for run in all_run_results:
        for combo_result in run.get('robust_combinations', []):
            # Combine IS and OOS metrics into a single row
            is_metrics = {k: v for k, v in combo_result['in_sample_metrics'].items() if k != 'combination'}
            oos_metrics = {'oos_' + k: v for k, v in combo_result['out_of_sample_metrics'].items()}
            flat_results.append({
                'symbol': run['symbol'], 
                'interval': run['interval'],
                'combination': ", ".join(map(str, sorted(combo_result['combination']))),
                **is_metrics,
                **oos_metrics
            })
    
    if flat_results:
        all_results_df = pd.DataFrame(flat_results)
        display_final_report(all_results_df)
    else:
        logging.warning("No robust combinations found across all runs. Cannot perform meta-analysis.")


if __name__ == '__main__':
    # freeze_support() is necessary for multiprocessing on Windows/macOS when bundled
    freeze_support()
    main()
