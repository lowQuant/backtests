import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import arcticdb as adb
except ImportError:
    adb = None
import dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 10_000_000  # 10M Yen (~$70k-100k depending on FX)
SIZES_TO_TEST = [0.01, 0.015, 0.02, 0.025, 0.03]
MAX_POSITIONS = 50 
HISTORY_FILE = "jp_stock_history_10y.csv"
PERIOD_YEARS_DOWNLOAD = "10y"
BACKTEST_YEARS = 10 

# Strategy Parameters
RSI_WINDOW = 4  
RSI_THRESHOLD = 90
SPIKE_WINDOW = 3
SPIKE_THRESHOLD = 0.15 
EMA_EXIT_WINDOW = 5
MIN_PRICE = 500.0        # 500 Yen (~$3.50)
MIN_VOLUME = 200_000    # Reduced from 2M for JP market liquidity

# Commission (Interactive Brokers JP Tiered approx)
COMMISSION_RATE = 0.0005 # 0.05% (5 bps)
MIN_COMMISSION = 100.0   # 100 Yen min

def load_arctic():
    """Load ArcticDB connection using environment variables."""
    if adb is None:
        raise ImportError("ArcticDB module not found.")
        
    dotenv.load_dotenv()
    bucket_name = os.getenv("BUCKET_NAME")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not bucket_name or not aws_access_key_id or not aws_secret_access_key or not aws_region:
        bucket_name = os.getenv("BUCKET_NAME2")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID2")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY2")
        
        if not bucket_name or not aws_access_key_id or not aws_secret_access_key:
            raise RuntimeError("Missing ArcticDB S3 environment variables.")

    conn_str = (
        f"s3://s3.{aws_region}.amazonaws.com:{bucket_name}"
        f"?region={aws_region}&access={aws_access_key_id}&secret={aws_secret_access_key}"
    )
    return adb.Arctic(conn_str)

def get_all_symbols():
    """Fetch all symbols from the CSV or Arctic."""
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / HISTORY_FILE
    if csv_path.exists():
        print(f"Local history file found at {csv_path}. Using symbols from there.")
        try:
            # Read just the header and unique symbols if possible, but reading whole csv is safer for structure
            # To be fast, let's just read columns or use the cached df logic later.
            # Here we just want list of symbols.
            df = pd.read_csv(csv_path, usecols=['Symbol'])
            symbols = sorted(df['Symbol'].unique().tolist())
            print(f"Found {len(symbols)} symbols in local file.")
            return symbols
        except Exception as e2:
            print(f"Error reading local file: {e2}")
    
    print("No local file found. Returning empty list (Arctic not implemented for JP yet in this script).")
    return []

def download_benchmark(start_date):
    """Download Nikkei 225 data for benchmark comparison."""
    print("Downloading Benchmark data (^N225)...")
    ticker = "^N225"
    try:
        spy = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if not spy.empty:
            # Handle MultiIndex columns if present (common in new yfinance)
            if isinstance(spy.columns, pd.MultiIndex):
                # Try to extract Cross-section for ticker
                try:
                    spy = spy.xs(ticker, axis=1, level=1)
                except KeyError:
                    # If ticker level not found, maybe it's just Price/Ticker hierarchy reversed or different
                    # Just checking if 'Close' is in level 0
                    if 'Close' in spy.columns.get_level_values(0):
                        spy.columns = spy.columns.get_level_values(0)
            
            # Ensure index is datetime and normalized
            if not isinstance(spy.index, pd.DatetimeIndex):
                spy.index = pd.to_datetime(spy.index)
            
            # Remove timezone if present to match backtest data
            if spy.index.tz is not None:
                spy.index = spy.index.tz_localize(None)
                
            spy.index = spy.index.normalize()
            
            # Handle duplicates
            if spy.index.duplicated().any():
                spy = spy.resample('D').last().dropna()

            if "Close" in spy.columns:
                print(f"Successfully downloaded {ticker} ({len(spy)} rows)")
                return spy, "Nikkei 225"
                
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
            
    return pd.DataFrame(), "None"

def download_and_cache_data(symbols):
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / HISTORY_FILE
    
    if csv_path.exists():
        print(f"Loading historical data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}.")
            return pd.DataFrame() # Return empty if fail, assuming file must exist for JP task

    print("Error: CSV file not found and download logic not configured for JP stocks in this script.")
    return pd.DataFrame()

def prepare_data(df):
    """
    Calculate indicators for Short Spike Strategy:
    1. 3-Day Returns
    2. RSI(4)
    3. EMA(5) for exit
    4. ADX(14) for Ranking
    """
    print("Calculating indicators...")
    df = df.sort_index()
    
    if "Close" not in df.columns or "Open" not in df.columns:
        df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)

    g = df.groupby(level='Symbol')

    # 1. 3-Day Return
    df['Ret_3D'] = g['Close'].pct_change(SPIKE_WINDOW)
    
    # 2. RSI (Wilder's)
    def calc_rsi_group(x):
        delta = x.diff()
        u = delta.clip(lower=0)
        d = -delta.clip(upper=0)
        # Wilder's Smoothing
        ewm_u = u.ewm(alpha=1/RSI_WINDOW, adjust=False).mean()
        ewm_d = d.ewm(alpha=1/RSI_WINDOW, adjust=False).mean()
        rs = ewm_u / ewm_d
        return 100 - (100 / (1 + rs))

    df['RSI'] = g['Close'].transform(calc_rsi_group)

    # 3. EMA 5
    df['EMA_Exit'] = g['Close'].transform(lambda x: x.ewm(span=EMA_EXIT_WINDOW, adjust=False).mean())
    
    # 4. ATR(5), ATR(10), ATR(20)
    h = df['High']
    l = df['Low']
    pc = g['Close'].shift(1)
    
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = g['TR'].transform(lambda x: x.rolling(window=5).mean())
    df['ATR_10'] = g['TR'].transform(lambda x: x.rolling(window=10).mean())
    df['ATR_20'] = g['TR'].transform(lambda x: x.rolling(window=20).mean())
    
    # 5. ADX (14) for Ranking
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    df['Plus_DM'] = plus_dm
    df['Minus_DM'] = minus_dm
    
    def calc_adx_group(x):
        n = 14
        tr_smooth = x['TR'].ewm(alpha=1/n, adjust=False).mean()
        plus_smooth = x['Plus_DM'].ewm(alpha=1/n, adjust=False).mean()
        minus_smooth = x['Minus_DM'].ewm(alpha=1/n, adjust=False).mean()
        
        plus_di = 100 * (plus_smooth / tr_smooth)
        minus_di = 100 * (minus_smooth / tr_smooth)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/n, adjust=False).mean()
        return adx
    
    df['ADX'] = g.apply(calc_adx_group, include_groups=False).reset_index(level=0, drop=True)

    # 6. Prev values for execution
    df['Prev_Close'] = pc
    df['Prev_EMA_Exit'] = g['EMA_Exit'].shift(1)
    df['Prev_ATR'] = g['ATR'].shift(1)
    df['Prev_ATR_10'] = g['ATR_10'].shift(1)
    df['Prev_ATR_20'] = g['ATR_20'].shift(1)
    
    return df

def run_simulation(df, trade_pct_equity, sl_config=None):
    sl_desc = "No Stop"
    if sl_config:
        sl_desc = sl_config['desc']
        
    print(f"\n>>> Simulating Strategy: Short Parabolic Spike JP (Size: {trade_pct_equity*100}%, SL: {sl_desc})")
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    # Entry Condition: Spike AND High RSI
    condition = (
        (test_df['Ret_3D'] > SPIKE_THRESHOLD) & 
        (test_df['RSI'] > RSI_THRESHOLD) &
        (test_df['Close'] >= MIN_PRICE) &
        (test_df['Volume'] >= MIN_VOLUME)
    )
    
    test_df['Signal'] = condition
    test_df['Pending_Short_Signal'] = test_df.groupby(level='Symbol')['Signal'].shift(1).fillna(False).infer_objects()
    
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    positions = [] 
    
    skipped_trades = 0
    # Structure: Reason -> [Wins, Losses]
    exit_counts = {
        "Stop Loss": [0, 0], 
        "Cross Under EMA5 (Target Met)": [0, 0], 
        "Time Stop": [0, 0]
    }
    
    for date in tqdm(sim_dates, desc="Simulating Days"):
        try:
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
            
        remaining_positions = []
        
        # 1. Manage Existing SHORT Positions
        for pos in positions:
            symbol = pos['Symbol']
            if symbol not in day_data.index:
                remaining_positions.append(pos)
                continue
                
            row = day_data.loc[symbol]
            open_price = row['Open']
            high = row['High']
            close = row['Close']
            ema_exit = row['EMA_Exit']
            
            exit_price = None
            exit_reason = None
            
            # 1. Stop Loss
            if pos['Stop_Price'] is not None and high >= pos['Stop_Price']:
                exit_price = max(open_price, pos['Stop_Price'])
                exit_reason = "Stop Loss"
            
            # 2. Target Exit
            if exit_price is None:
                if close < ema_exit:
                    exit_price = close
                    exit_reason = "Cross Under EMA5 (Target Met)"
            
            # 3. Time Stop
            pos['Days_Held'] += 1
            if exit_price is None and pos['Days_Held'] >= 10:
                exit_price = close
                exit_reason = "Time Stop"
                
            if exit_price is not None:
                gross_pnl = (pos['Entry_Price'] - exit_price) * pos['Shares']
                
                # Commission Calculation (JP Value Based)
                exit_value = exit_price * pos['Shares']
                comm_exit = max(MIN_COMMISSION, exit_value * COMMISSION_RATE)
                
                net_pnl = gross_pnl - comm_exit - pos['Commission_Entry']
                
                cash += (pos['Entry_Price'] * pos['Shares']) + net_pnl
                
                if exit_reason in exit_counts:
                    if net_pnl > 0:
                        exit_counts[exit_reason][0] += 1
                    else:
                        exit_counts[exit_reason][1] += 1
                
                trades.append({
                    'Symbol': symbol,
                    'Entry_Date': pos['Entry_Date'],
                    'Exit_Date': date,
                    'Entry_Price': pos['Entry_Price'],
                    'Exit_Price': exit_price,
                    'Return': (pos['Entry_Price'] - exit_price) / pos['Entry_Price'],
                    'PnL': net_pnl,
                    'Reason': exit_reason,
                    'Days_Held': pos['Days_Held'],
                    'Comm_Entry': pos['Commission_Entry'],
                    'Comm_Exit': comm_exit
                })
            else:
                remaining_positions.append(pos)
                
        positions = remaining_positions
        
        # 2. Process New Entries (SHORTS)
        if len(positions) < MAX_POSITIONS:
            candidates = day_data[day_data['Pending_Short_Signal']].copy()
            # Ranking: High ADX -> Strongest Trend First
            candidates = candidates.sort_values('ADX', ascending=False)
            
            for symbol, row in candidates.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    skipped_trades += 1 
                    continue
                
                if any(p['Symbol'] == symbol for p in positions):
                    continue
                
                # Market Entry on Open
                entry_price = row['Open']
                
                # Calculate Stop Price
                stop_price = None
                if sl_config:
                    if sl_config['type'] == 'ATR':
                        atr_period = sl_config.get('atr_period', 20)
                        if atr_period == 10:
                            atr_val = row['Prev_ATR_10']
                        else:
                            atr_val = row['Prev_ATR_20']
                            
                        stop_price = entry_price + (sl_config['value'] * atr_val)
                    elif sl_config['type'] == 'PCT':
                        stop_price = entry_price * (1.0 + sl_config['value'])
                
                # Sizing
                curr_eq = cash
                for p in positions:
                    s_sym = p['Symbol']
                    if s_sym in day_data.index:
                        curr_p = day_data.loc[s_sym]['Close']
                    else:
                        curr_p = p['Entry_Price']
                    
                    unrealized = (p['Entry_Price'] - curr_p) * p['Shares']
                    curr_eq += (p['Entry_Price'] * p['Shares']) + unrealized

                target_allocation = curr_eq * trade_pct_equity
                
                if target_allocation > cash:
                    target_allocation = cash
                
                shares = int(target_allocation / entry_price)
                
                if shares > 0:
                    entry_value = shares * entry_price
                    comm_entry = max(MIN_COMMISSION, entry_value * COMMISSION_RATE)
                    
                    if (entry_value + comm_entry) > cash:
                         # Adjust shares to fit cash including commission
                         # cash = shares * price + max(min, shares * price * rate)
                         # Simple approximation:
                         shares = int((cash - MIN_COMMISSION) / (entry_price * (1 + COMMISSION_RATE)))
                         entry_value = shares * entry_price
                         comm_entry = max(MIN_COMMISSION, entry_value * COMMISSION_RATE)
                    
                    if shares > 0:
                        cash -= (entry_value + comm_entry)
                        positions.append({
                            'Symbol': symbol,
                            'Entry_Price': entry_price,
                            'Shares': shares,
                            'Entry_Date': date,
                            'Days_Held': 0,
                            'Commission_Entry': comm_entry,
                            'Stop_Price': stop_price
                        })
        else:
            skipped_trades += len(day_data[day_data['Pending_Short_Signal']])
        
        # Calculate Daily Stats
        curr_eq = cash
        pos_value = 0 
        
        for p in positions:
            s_sym = p['Symbol']
            if s_sym in day_data.index:
                curr_p = day_data.loc[s_sym]['Close']
            else:
                curr_p = p['Entry_Price']
            
            unrealized = (p['Entry_Price'] - curr_p) * p['Shares']
            curr_eq += (p['Entry_Price'] * p['Shares']) + unrealized
            pos_value += curr_p * p['Shares'] 

        equity_curve.append({
            'Date': date,
            'Equity': curr_eq,
            'Cash': cash,
            'Short_Exposure': pos_value,
            'Num_Positions': len(positions)
        })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades), skipped_trades, exit_counts

def calculate_cagr(final_equity, years):
    if final_equity <= 0: return -1.0
    return (final_equity / STARTING_CAPITAL) ** (1 / years) - 1

def main():
    symbols = get_all_symbols()
    if not symbols:
        print("No symbols found.")
        return

    df = download_and_cache_data(symbols)
    if df.empty:
        print("DF is empty. Exiting.")
        return
        
    df = prepare_data(df)
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    spy, benchmark_ticker = download_benchmark(start_date)
    
    script_dir = Path(__file__).resolve().parent
    best_eq_df = None
    best_stats = None

    SCENARIOS = []
    # Add Size Sweep
    for size in SIZES_TO_TEST:
        SCENARIOS.append({
            'label': f"Size {size*100}%",
            'size': size,
            'sl_config': None
        })
    
    # Add Special SL Case
    SCENARIOS.append({
        'label': "Size 3% (SL 3*ATR10)",
        'size': 0.03,
        'sl_config': {'type': 'ATR', 'value': 3.0, 'desc': '3*ATR10', 'atr_period': 10}
    })

    print("\n" + "="*80)
    print(f"RUNNING JP STRATEGY SCENARIOS (Ranking: ADX, Capital: 10M JPY)")
    print("="*80)
    
    results = []
    
    for scen in SCENARIOS:
        eq_df, trades_df, skipped, exits = run_simulation(df, scen['size'], sl_config=scen['sl_config'])
        
        if eq_df.empty:
            continue
            
        final_equity = eq_df.iloc[-1]['Equity']
        total_ret = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL
        
        eq_df['Daily_Ret'] = eq_df['Equity'].pct_change()
        sharpe = (eq_df['Daily_Ret'].mean() / eq_df['Daily_Ret'].std()) * np.sqrt(252) if eq_df['Daily_Ret'].std() > 0 else 0
        
        eq_df['Peak'] = eq_df['Equity'].cummax()
        eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']
        max_dd = eq_df['Drawdown'].min()
        
        cagr = calculate_cagr(final_equity, BACKTEST_YEARS)
        
        # Correlation
        correlation = 0
        if not spy.empty:
             temp_df = eq_df.set_index('Date')
             active_days = temp_df[temp_df['Num_Positions'] > 0].index
             strat_ret_series = temp_df.loc[active_days, 'Daily_Ret']
             
             # Ensure spy['Close'] is a Series
             spy_close = spy['Close']
             if isinstance(spy_close, pd.DataFrame):
                 spy_close = spy_close.iloc[:, 0]
                 
             spy_ret_series = spy_close.pct_change()
             
             common_idx = strat_ret_series.index.intersection(spy_ret_series.index)
             if len(common_idx) > 10:
                 try:
                     correlation = strat_ret_series.loc[common_idx].corr(spy_ret_series.loc[common_idx])
                 except Exception:
                     correlation = 0
        
        # Trade Stats
        sqn = 0
        profit_factor = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        
        if not trades_df.empty:
            trade_rets = trades_df['Return']
            if trade_rets.std() > 0:
                sqn = np.sqrt(len(trades_df)) * (trade_rets.mean() / trade_rets.std())
            
            gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
            gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf')
                
            win_rate = (len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)) * 100
            avg_win = trades_df[trades_df['PnL'] > 0]['Return'].mean() * 100 if not trades_df[trades_df['PnL'] > 0].empty else 0
            avg_loss = trades_df[trades_df['PnL'] < 0]['Return'].mean() * 100 if not trades_df[trades_df['PnL'] < 0].empty else 0
        
        stats = {
            "Label": scen['label'],
            "CAGR %": cagr * 100,
            "Total Ret %": total_ret * 100,
            "Sharpe": sharpe,
            "Max DD %": max_dd * 100,
            "Corr Nikkei": correlation,
            "SQN": sqn,
            "Prof Fact": profit_factor,
            "Win Rate %": win_rate,
            "Avg Win %": avg_win,
            "Avg Loss %": avg_loss,
            "Trades": len(trades_df),
            "Skipped": skipped,
            "Stops": sum(exits["Stop Loss"]),
            "Tgt Wins": exits["Cross Under EMA5 (Target Met)"][0],
            "Tgt Loss": exits["Cross Under EMA5 (Target Met)"][1],
            "Time Wins": exits["Time Stop"][0],
            "Time Loss": exits["Time Stop"][1],
            "Final Eq": final_equity
        }
        results.append(stats)
        
        if best_stats is None or cagr > best_stats['CAGR %'] / 100:
            best_stats = stats
            best_eq_df = eq_df
            
    # Print Table
    res_df = pd.DataFrame(results)
    print("\n" + "="*160)
    print("SHORT STRATEGY RESULTS JP (Sorted by ADX)")
    print("="*160)
    cols = ["Label", "CAGR %", "Total Ret %", "Sharpe", "Max DD %", "Prof Fact", "Win Rate %", "Avg Win %", "Avg Loss %", "Trades", "Skipped", "Stops", "Tgt Wins", "Tgt Loss", "Time Wins", "Time Loss"]
    print(res_df[cols].to_string(index=False, formatters={
        "CAGR %": "{:.2f}".format,
        "Total Ret %": "{:.2f}".format,
        "Sharpe": "{:.2f}".format,
        "Max DD %": "{:.2f}".format,
        "Prof Fact": "{:.2f}".format,
        "Win Rate %": "{:.2f}".format,
        "Avg Win %": "{:.2f}".format,
        "Avg Loss %": "{:.2f}".format
    }))
    print("="*160)
    
    # Plotting Best Run
    if best_eq_df is not None:
        best_label = best_stats['Label']
        print(f"\nPlotting results for best run: {best_label}")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. Equity Curve
        spy_aligned = spy[spy.index >= best_eq_df.iloc[0]['Date']].copy()
        if not spy_aligned.empty:
            spy_aligned['Norm'] = (spy_aligned['Close'] / spy_aligned.iloc[0]['Close']) * STARTING_CAPITAL
            ax1.plot(spy_aligned.index, spy_aligned['Norm'], label=f"{benchmark_ticker} (Benchmark)", color='gray', alpha=0.6, linestyle='--')
            
        ax1.plot(best_eq_df['Date'], best_eq_df['Equity'], label=f"Short Strategy ({best_label})", color='red')
        ax1.set_ylabel("Equity (JPY)")
        ax1.set_title(f"Short Parabolic Spike JP ({best_label}) vs {benchmark_ticker}\nCAGR: {best_stats['CAGR %']:.2f}% | Max DD: {best_stats['Max DD %']:.2f}% | Sharpe: {best_stats['Sharpe']:.2f} | PF: {best_stats['Prof Fact']:.2f} | Win Rate: {best_stats['Win Rate %']:.2f}%")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Exposure
        ax2.plot(best_eq_df['Date'], best_eq_df['Num_Positions'], color='purple', label="Num Shorts")
        ax2.set_ylabel("Positions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3.fill_between(best_eq_df['Date'], best_eq_df['Drawdown'] * 100, 0, color='red', alpha=0.3, label="Drawdown %")
        ax3.set_ylabel("Drawdown %")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(script_dir / "b_short_spike_jp_summary.png")
        print(f"Comparison plot saved to {script_dir / 'b_short_spike_jp_summary.png'}")

if __name__ == "__main__":
    main()
