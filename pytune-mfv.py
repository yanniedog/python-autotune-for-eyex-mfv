import os
import sys
import argparse
import time
import json
import logging
import subprocess
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CONFIG = {
    'BASE_URL': 'https://api.binance.com',
    'COMPREHENSIVE_SYMBOLS': ['SOLUSDT', 'PAXGUSDT'],
    'COMPREHENSIVE_INTERVALS': ['1s', '1m', '5m', '1h', '4h', '1d'],
    'MAX_BARS_TO_DOWNLOAD': 20000,
    'ANALYSIS_TIMEOUT_SECONDS': 120 # 2-minute timeout for each analysis
}

# ==============================================================================
# --- LOGGING SETUP ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mfv_orchestrator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# --- DATA FETCHING ---
# ==============================================================================
session = requests.Session()
session.headers.update({'Accept-Encoding': 'gzip'})

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_request(url, params=None):
    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def get_historical_klines(symbol, interval):
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
                if not data: break
                klines.extend(data)
                pbar.update(len(data))
                end_time = data[0][0] - 1
                if len(data) < fetch_limit: break
            except Exception as e:
                logging.error(f"Error fetching klines for {symbol}/{interval}: {e}")
                return None
    
    return sorted(klines, key=lambda x: x[0])

def process_and_save_klines(klines, symbol, interval):
    if not klines: return None
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbav', 'tbqv', 'ignore'])
    df = df[cols[:6]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    filepath = f"temp_data_{symbol}_{interval}.csv"
    df.to_csv(filepath, index=False)
    logging.info(f"Processed {len(df)} klines and saved to {filepath}")
    return filepath

# ==============================================================================
# --- REPORTING ---
# ==============================================================================
def display_comprehensive_report(all_results_df):
    if all_results_df.empty:
        logging.warning("No valid results found across all runs for meta-analysis.")
        return

    df = all_results_df.copy()
    
    # Normalize metrics for fair scoring
    df['youdens_j_norm'] = (df['youdens_j'] - df['youdens_j'].min()) / (df['youdens_j'].max() - df['youdens_j'].min())
    df['calmar_ratio_norm'] = (df['calmar_ratio'] - df['calmar_ratio'].min()) / (df['calmar_ratio'].max() - df['calmar_ratio'].min())
    df['max_drawdown_pct_inv_norm'] = 1 - ((df['max_drawdown_pct'] - df['max_drawdown_pct'].min()) / (df['max_drawdown_pct'].max() - df['max_drawdown_pct'].min()))
    
    # Calculate robustness score with 50% weight on Youden's J
    df['robustness_score'] = 0.5 * df['youdens_j_norm'].fillna(0) + 0.3 * df['calmar_ratio_norm'].fillna(0) + 0.2 * df['max_drawdown_pct_inv_norm'].fillna(0)
    df = df.sort_values('robustness_score', ascending=False)
    
    print("\n" + "#"*170 + "\n" + "--- COMPREHENSIVE WALK-FORWARD META-ANALYSIS REPORT ---".center(170) + "\n" + "#"*170)
    header = f"{'Rank':<5} | {'Symbol':<10} | {'Interval':<8} | {'Robust Score':<14} | {'Youden\'s J':<12} | {'Calmar Ratio':<14} | {'Max DD %':<12} | {'Win Rate':<10} | {'Best Combination'}"
    print(header + "\n" + "-" * len(header))
    for i, row in enumerate(df.head(20).itertuples(), 1):
        robust_score = f"{row.robustness_score:.3f}"
        youdens_j = f"{row.youdens_j:.3f}"
        calmar = f"{row.calmar_ratio:.2f}" if np.isfinite(row.calmar_ratio) else "inf"
        max_dd = f"{row.max_drawdown_pct:.2f}%"
        win_rate = f"{row.win_rate:.1%}"
        print(f"{i:<5} | {row.symbol:<10} | {row.interval:<8} | {robust_score:<14} | {youdens_j:<12} | {calmar:<14} | {max_dd:<12} | {win_rate:<10} | {row.combination}")
    print("#"*170)

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Walk-Forward MFV Analysis.")
    parser.add_argument(
        'mode', default='comprehensive', choices=['comprehensive'],
        help="Mode of operation."
    )
    args = parser.parse_args()

    if args.mode == 'comprehensive':
        all_run_results = []
        analyzer_script_path = "walk_forward_analyzer.py"

        if not os.path.exists(analyzer_script_path):
            logging.error(f"FATAL: The analyzer script '{analyzer_script_path}' was not found in the same directory.")
            return

        for symbol in CONFIG['COMPREHENSIVE_SYMBOLS']:
            for interval in CONFIG['COMPREHENSIVE_INTERVALS']:
                logging.info(f"--- STARTING WALK-FORWARD ANALYSIS FOR: {symbol} | {interval} ---")
                
                # 1. Fetch data and save to a temporary file
                klines = get_historical_klines(symbol, interval)
                temp_data_path = process_and_save_klines(klines, symbol, interval)
                
                if not temp_data_path:
                    logging.warning(f"Skipping {symbol}/{interval} due to data fetching/processing error.")
                    continue

                # 2. Run the analyzer script as a subprocess with a timeout
                try:
                    command = [sys.executable, analyzer_script_path, temp_data_path, symbol, interval]
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=CONFIG['ANALYSIS_TIMEOUT_SECONDS']
                    )
                    
                    # 3. Parse the JSON result from the subprocess
                    output = json.loads(result.stdout)
                    if output.get("status") == "success":
                        all_run_results.append(output['data'])
                    else:
                        logging.error(f"Analysis failed for {symbol}/{interval}. Reason: {output.get('error', 'Unknown')}")

                except subprocess.TimeoutExpired:
                    logging.error(f"Analysis for {symbol}/{interval} timed out after {CONFIG['ANALYSIS_TIMEOUT_SECONDS']} seconds. Skipping.")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error executing analyzer for {symbol}/{interval}:")
                    logging.error(e.stderr)
                except json.JSONDecodeError:
                    logging.error(f"Could not decode JSON from analyzer for {symbol}/{interval}.")
                finally:
                    # 4. Clean up the temporary file
                    if os.path.exists(temp_data_path):
                        os.remove(temp_data_path)

        # 5. Generate the final report
        if all_run_results:
            final_df = pd.DataFrame(all_run_results)
            display_comprehensive_report(final_df)
            final_df.to_csv("walk_forward_report.csv", index=False)
            logging.info("Comprehensive walk-forward report saved to 'walk_forward_report.csv'")
        else:
            logging.warning("No successful analysis runs to report.")

if __name__ == '__main__':
    main()
