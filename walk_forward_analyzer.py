import sys
import json
import logging
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

# ==============================================================================
# --- ADAPTIVE CONFIGURATION (FOR SPEED) ---
# ==============================================================================
# This is crucial to meet the 2-minute timeout for high-frequency data.
def get_config_for_interval(interval_str):
    base_config = {
        'FINAL_PEAK_COUNT': 12, 'COMBINATION_SIZE': 4,
        'STOP_LOSS_PCT': 5.0, 'TAKE_PROFIT_PCT': 15.0,
        'MIN_PEAK_DISTANCE': 50,
        # Walk-Forward Parameters
        'NUM_FOLDS': 5, # Number of walk-forward steps
        'TRAIN_TEST_RATIO': 3 # Train on 3 parts, test on 1 part
    }
    if 's' in interval_str:
        base_config.update({'BROAD_SCAN_START': 10, 'BROAD_SCAN_END': 1200, 'BROAD_SCAN_STEP': 10})
    elif 'm' in interval_str:
        base_config.update({'BROAD_SCAN_START': 10, 'BROAD_SCAN_END': 3000, 'BROAD_SCAN_STEP': 5})
    else: # Hours and Daily
        base_config.update({'BROAD_SCAN_START': 10, 'BROAD_SCAN_END': 5000, 'BROAD_SCAN_STEP': 5})
    return base_config

# ==============================================================================
# --- LOGGING ---
# ==============================================================================
# Note: This script's logs will be captured by the orchestrator's stderr.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

# ==============================================================================
# --- NUMBA ACCELERATED BACKTESTING ENGINE ---
# ==============================================================================
@jit(nopython=True)
def _numba_backtest_loop(open_price, high_price, low_price, signal, horizon, sl_pct, tp_pct):
    returns = np.zeros(len(open_price))
    trade_count = 0
    for i in range(1, len(open_price)):
        if signal[i-1] != 0:
            entry_price, trade_direction, outcome = open_price[i], np.sign(signal[i-1]), 0.0
            if trade_direction > 0:
                sl_price, tp_price = entry_price * (1 - sl_pct / 100), entry_price * (1 + tp_pct / 100)
                for k in range(i, min(i + horizon, len(open_price))):
                    if low_price[k] <= sl_price: outcome = -sl_pct; break
                    if high_price[k] >= tp_price: outcome = tp_pct; break
                if outcome == 0.0:
                    exit_idx = min(i + horizon, len(open_price) - 1)
                    outcome = (open_price[exit_idx] / entry_price - 1) * 100
            returns[trade_count] = outcome
            trade_count += 1
    return returns[:trade_count]

# ==============================================================================
# --- ANALYSIS & METRICS ---
# ==============================================================================
def get_periods_per_year(interval_str):
    if 'm' in interval_str: return 365 * 24 * (60 / int(interval_str.replace('m', '')))
    if 'h' in interval_str: return 365 * (24 / int(interval_str.replace('h', '')))
    if 'd' in interval_str: return 365
    if 's' in interval_str: return 365 * 24 * 60 * (60 / int(interval_str.replace('s', '')))
    return 252

def calculate_final_metrics(all_returns, num_total_bars, interval):
    if not all_returns: return {}
    returns_np = np.concatenate(all_returns)
    if returns_np.size == 0: return {}
    
    win_rate = np.sum(returns_np > 0) / len(returns_np)
    equity_curve = np.cumsum(returns_np)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    max_drawdown_pct = np.max(drawdown)
    
    periods_per_year = get_periods_per_year(interval)
    num_years = num_total_bars / periods_per_year
    total_return = equity_curve[-1]
    annualized_return = (total_return / num_years) if num_years > 0 else 0
    calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else np.inf

    # Youden's J isn't calculated on the aggregate, as it's a training metric.
    # We can use an average of the IS Youden's J if needed, but for now we focus on OOS performance.
    return {
        'win_rate': win_rate,
        'max_drawdown_pct': max_drawdown_pct,
        'calmar_ratio': calmar_ratio,
        'num_trades': len(returns_np)
    }

def run_optimization_on_fold(df_train, interval, config):
    """This function runs the full optimization on a single training fold."""
    # 1. Broad Scan
    start, end, step = config['BROAD_SCAN_START'], config['BROAD_SCAN_END'], config['BROAD_SCAN_STEP']
    close, high, low, volume = df_train['close'], df_train['high'], df_train['low'], df_train['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan); mf_multiplier.fillna(0, inplace=True)
    mf_volume = mf_multiplier * volume
    scan_results = []
    for period in range(start, end + 1, step):
        if len(df_train) < period * 2: continue
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
        scan_results.append({'period': period, 'youdens_j': sensitivity + specificity - 1})

    if not scan_results: return None

    # 2. Peak Selection
    sorted_by_j = sorted(scan_results, key=lambda x: x['youdens_j'], reverse=True)
    diverse_peaks = []
    if sorted_by_j:
        diverse_peaks.append(sorted_by_j[0])
        for peak in sorted_by_j[1:]:
            if len(diverse_peaks) >= config['FINAL_PEAK_COUNT']: break
            if all(abs(peak['period'] - p['period']) >= config['MIN_PEAK_DISTANCE'] for p in diverse_peaks):
                diverse_peaks.append(peak)
    
    if len(diverse_peaks) < config['COMBINATION_SIZE']: return None

    # 3. Combination Analysis
    peak_combinations = list(combinations([p['period'] for p in diverse_peaks], config['COMBINATION_SIZE']))
    best_combo_for_fold = None
    best_j_for_fold = -np.inf

    for combo in peak_combinations:
        mf_volume_train = (((df_train['close'] - df_train['low']) - (df_train['high'] - df_train['close'])) / (df_train['high'] - df_train['low']).replace(0, np.nan)).fillna(0) * df_train['volume']
        def normalize_zscore(data, window):
            mean = data.rolling(window=window).mean(); stdev = data.rolling(window=window).std().replace(0, np.nan)
            return ((data - mean) / stdev).fillna(0)
        mfv_lines = [(normalize_zscore(mf_volume_train.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in combo]
        composite_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)
        
        min_horizon, max_horizon = min(combo), int(max(combo) * 1.2)
        for horizon in range(min_horizon, max_horizon + 1, config['BROAD_SCAN_STEP']):
            if len(df_train) < horizon + max(combo): continue
            
            price_roc = df_train['close'].pct_change(periods=horizon).shift(-horizon) * 100
            combined = pd.DataFrame({'signal': composite_signal, 'roc': price_roc}).dropna()
            signal_sign, roc_sign = np.sign(combined['signal']), np.sign(combined['roc'])
            tp = np.sum((signal_sign > 0) & (roc_sign > 0)); fn = np.sum((signal_sign <= 0) & (roc_sign > 0))
            tn = np.sum((signal_sign < 0) & (roc_sign < 0)); fp = np.sum((signal_sign >= 0) & (roc_sign < 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0; specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youdens_j = sensitivity + specificity - 1

            if youdens_j > best_j_for_fold:
                best_j_for_fold = youdens_j
                best_combo_for_fold = {'combination': combo, 'horizon': horizon, 'youdens_j': youdens_j}

    return best_combo_for_fold

# ==============================================================================
# --- MAIN WORKER ---
# ==============================================================================
def run_walk_forward(data_path, symbol, interval):
    try:
        df = pd.read_csv(data_path)
        config = get_config_for_interval(interval)
        
        fold_size = len(df) // (config['NUM_FOLDS'] + config['TRAIN_TEST_RATIO'] - 1)
        train_size = fold_size * config['TRAIN_TEST_RATIO']
        
        all_oos_returns = []
        best_params_per_fold = []

        for i in range(config['NUM_FOLDS']):
            logging.info(f"Running Fold {i+1}/{config['NUM_FOLDS']}...")
            start_idx = i * fold_size
            train_end_idx = start_idx + train_size
            test_end_idx = train_end_idx + fold_size
            
            if test_end_idx > len(df): break

            df_train = df.iloc[start_idx:train_end_idx].copy()
            df_test = df.iloc[train_end_idx:test_end_idx].copy()

            # Find best parameters on the training fold
            best_params = run_optimization_on_fold(df_train, interval, config)
            if not best_params:
                logging.warning(f"No valid parameters found for fold {i+1}.")
                continue
            
            best_params_per_fold.append(best_params)

            # Apply best parameters to the out-of-sample (test) fold
            combo, horizon = best_params['combination'], best_params['horizon']
            mf_volume_test = (((df_test['close'] - df_test['low']) - (df_test['high'] - df_test['close'])) / (df_test['high'] - df_test['low']).replace(0, np.nan)).fillna(0) * df_test['volume']
            def normalize_zscore(data, window):
                mean = data.rolling(window=window).mean(); stdev = data.rolling(window=window).std().replace(0, np.nan)
                return ((data - mean) / stdev).fillna(0)
            mfv_lines = [(normalize_zscore(mf_volume_test.rolling(window=r).sum(), r) * 10).clip(-100, 100) for r in combo]
            oos_signal = pd.concat(mfv_lines, axis=1).mean(axis=1)

            _, oos_returns = calculate_performance_metrics(df_test, oos_signal, horizon, config['STOP_LOSS_PCT'], config['TAKE_PROFIT_PCT'], interval)
            all_oos_returns.append(oos_returns)

        # Aggregate results
        if not best_params_per_fold:
            raise ValueError("Walk-forward analysis failed to find any robust parameters.")

        final_metrics = calculate_final_metrics(all_oos_returns, len(df), interval)
        
        # Find the most frequently occurring "best" combination
        combo_counts = pd.Series([str(sorted(p['combination'])) for p in best_params_per_fold]).value_counts()
        best_overall_combo_str = combo_counts.index[0]
        
        # Average the Youden's J score from the training folds
        avg_youdens_j = np.mean([p['youdens_j'] for p in best_params_per_fold])

        result = {
            'symbol': symbol,
            'interval': interval,
            'combination': best_overall_combo_str,
            'youdens_j': avg_youdens_j,
            **final_metrics
        }
        
        print(json.dumps({"status": "success", "data": result}))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(json.dumps({"status": "error", "error": "Invalid arguments. Usage: script.py <data_path> <symbol> <interval>"}))
        sys.exit(1)
    
    run_walk_forward(sys.argv[1], sys.argv[2], sys.argv[3])
