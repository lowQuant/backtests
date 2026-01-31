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
STARTING_CAPITAL = 100_000
SIZES_TO_TEST = [0.01, 0.03, 0.05, 0.08, 0.10]
MAX_POSITIONS = 100 
HISTORY_FILE = "us_stock_history_10y.csv"
PERIOD_YEARS_DOWNLOAD = "10y"
BACKTEST_YEARS = 10 
EMA_WINDOW = 200
ATR_WINDOW = 5
DROP_THRESHOLD = -0.03
ATR_ENTRY_MULT = 0.9
ATR_TARGET_MULT = 0.5
TIME_STOP_DAYS = 10
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0
MIN_PRICE = 20.0        
MIN_VOLUME = 5_000_000  

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
    """Fetch all symbols from the ALL_STOCKS table in ArcticDB."""
    # First check if we have local history file, if so we can just return a dummy list 
    # because download_and_cache_data will read from the file anyway.
    # However, download_and_cache_data uses symbols list to download if file missing.
    # Let's try ArcticDB first.
    
    print("Connecting to ArcticDB to fetch symbol list...")
    try:
        ac = load_arctic()
        lib = ac.get_library("us_equities")
        all_stocks = lib.read("ALL_STOCKS").data
        
        if "Symbol" in all_stocks.columns:
            symbols = sorted(all_stocks["Symbol"].unique().tolist())
        elif "Ticker" in all_stocks.columns:
            symbols = sorted(all_stocks["Ticker"].unique().tolist())
        else:
            symbols = sorted(all_stocks.index.unique().tolist())
            
        print(f"Found {len(symbols)} symbols from ArcticDB.")
        return symbols
        
    except Exception as e:
        print(f"ArcticDB unavailable: {e}")
        
        # Check if local file exists, if so, we can rely on it
        script_dir = Path(__file__).resolve().parent
        csv_path = script_dir / HISTORY_FILE
        if csv_path.exists():
            print(f"Local history file found at {csv_path}. Using symbols from there.")
            try:
                # Read just the header or index to get symbols? 
                # Actually download_and_cache_data reads the whole file. 
                # We need to return SOME list to proceed.
                # Let's read the index level 1 (Symbol)
                df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
                symbols = df.index.get_level_values(1).unique().tolist()
                print(f"Found {len(symbols)} symbols in local file.")
                return symbols
            except Exception as e2:
                print(f"Error reading local file: {e2}")

        print("Falling back to default symbol list.")
        return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL"]

def download_spy(start_date, df_universe=None):
    """Download SPY data for benchmark comparison."""
    print("Downloading Benchmark data...")
    
    # 1. Try Local Parquet (Highest Priority)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    parquet_files = list(data_dir.glob("SPY_*.parquet"))
    
    if parquet_files:
        # Select the file with the longest duration based on filename timestamps
        # Format: SPY_YYYYMMDD_YYYYMMDD.parquet
        best_file = None
        max_duration = -1
        
        for f in parquet_files:
            try:
                # Extract dates from filename
                parts = f.stem.split('_')
                if len(parts) >= 3:
                    start_str, end_str = parts[1], parts[2]
                    start_dt = pd.to_datetime(start_str)
                    end_dt = pd.to_datetime(end_str)
                    duration = (end_dt - start_dt).days
                    
                    if duration > max_duration:
                        max_duration = duration
                        best_file = f
            except Exception:
                continue
        
        # Fallback to file size if parsing fails
        if best_file is None:
             best_file = max(parquet_files, key=lambda f: f.stat().st_size)
             
        print(f"Loading local benchmark from {best_file} (Duration: {max_duration} days)...")
        
        try:
            spy = pd.read_parquet(best_file)
            # Rename columns to Title Case if necessary
            spy.rename(columns=lambda x: x.capitalize(), inplace=True)
            
            # Ensure index is datetime
            if not isinstance(spy.index, pd.DatetimeIndex):
                spy.index = pd.to_datetime(spy.index)
            
            # Normalize to Midnight (remove time component) for daily merging
            spy.index = spy.index.normalize()
            
            # Handle duplicates (if intraday data was loaded, take the last close of the day)
            if spy.index.duplicated().any():
                print("Aggregating intraday benchmark data to daily...")
                spy = spy.resample('D').last().dropna()
            
            # Filter start date
            spy = spy[spy.index >= start_date]
            
            if not spy.empty and "Close" in spy.columns:
                print(f"Successfully loaded local SPY data ({len(spy)} rows)")
                return spy, "SPY"
        except Exception as e:
            print(f"Error loading local parquet: {e}")

    # 2. Try Yahoo Finance
    tickers_to_try = ["^GSPC", "SPY", "^SPY"] 
    
    for ticker in tickers_to_try:
        try:
            print(f"Attempting to download {ticker}...")
            spy = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if not spy.empty:
                if "Close" not in spy.columns:
                     if isinstance(spy.columns, pd.MultiIndex):
                         # Handle MultiIndex: try to find the ticker level
                         try:
                            spy = spy.xs(ticker, axis=1, level=1)
                         except KeyError:
                             pass
                
                if "Close" in spy.columns:
                    print(f"Successfully downloaded {ticker}")
                    return spy, ticker
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            
    print("Failed to download any benchmark data via yfinance.")
    
    # Fallback: Synthetic Universe Index
    if df_universe is not None:
        print("Constructing Synthetic Equal-Weight Universe Index as fallback...")
        try:
            # Ensure Daily_Ret is calculated
            if 'Daily_Ret' not in df_universe.columns:
                 print("Daily_Ret missing, cannot construct synthetic index.")
                 return pd.DataFrame(), "None"
                 
            daily_ret = df_universe.groupby(level='Date')['Daily_Ret'].mean()
            # Filter to start_date
            daily_ret = daily_ret[daily_ret.index >= start_date]
            
            if not daily_ret.empty:
                # Reconstruct price (Start at 100)
                # We need cumulative product
                # Fill NaN with 0 for cumprod
                price_series = (1 + daily_ret.fillna(0)).cumprod() * 100
                
                bench_df = pd.DataFrame({'Close': price_series})
                return bench_df, "Synthetic Universe Index"
        except Exception as e:
            print(f"Error constructing synthetic index: {e}")

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
            print(f"Error loading CSV file: {e}. Re-downloading.")

    print(f"Downloading {PERIOD_YEARS_DOWNLOAD} of data for {len(symbols)} symbols...")
    
    chunk_size = 500
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    all_dfs = []

    for chunk in tqdm(chunks, desc="Downloading chunks"):
        try:
            data = yf.download(
                chunk,
                period=PERIOD_YEARS_DOWNLOAD,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            if data.empty:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                stacked = data.stack(level=0)
                stacked.index.names = ['Date', 'Symbol']
                all_dfs.append(stacked)
            else:
                data['Symbol'] = chunk[0]
                data = data.set_index('Symbol', append=True)
                all_dfs.append(data)
                
        except Exception as e:
            print(f"Error downloading chunk: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data downloaded.")

    full_df = pd.concat(all_dfs).sort_index()
    full_df.to_csv(csv_path)
    return full_df

def prepare_data(df):
    """
    Calculate indicators:
    1. 200 EMA
    2. ATR(5)
    3. Daily Returns
    """
    print("Calculating indicators...")
    df = df.sort_index()
    
    if "Close" not in df.columns or "Open" not in df.columns:
        df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)

    g = df.groupby(level='Symbol')

    # 1. EMA 200
    df['EMA_200'] = g['Close'].transform(lambda x: x.ewm(span=EMA_WINDOW, adjust=False).mean())

    # 2. Daily Returns
    df['Prev_Close'] = g['Close'].shift(1)
    df['Daily_Ret'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']

    # 3. ATR(5)
    h = df['High']
    l = df['Low']
    pc = df['Prev_Close']
    
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_5'] = g['TR'].transform(lambda x: x.rolling(window=ATR_WINDOW).mean())
    
    # 4. Previous High (for exit condition)
    df['Prev_High'] = g['High'].shift(1)

    return df

def run_simulation(df, trade_pct_equity):
    print(f"\n>>> Simulating Strategy: Panic Dip + ATR Target (Size: {trade_pct_equity*100}%)")
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    condition = (
        (test_df['Close'] > test_df['EMA_200']) & 
        (test_df['Daily_Ret'] < DROP_THRESHOLD) &
        (test_df['ATR_5'].notna()) &
        (test_df['Close'] >= MIN_PRICE) &
        (test_df['Volume'] >= MIN_VOLUME)
    )
    
    test_df['Signal'] = condition
    
    test_df['Setup_Close'] = test_df['Close']
    test_df['Setup_ATR'] = test_df['ATR_5']
    test_df['Limit_Entry_Price'] = test_df['Close'] - (ATR_ENTRY_MULT * test_df['ATR_5'])
    test_df['Target_Exit_Price'] = test_df['Close'] + (ATR_TARGET_MULT * test_df['ATR_5'])
    
    test_df['Pending_Buy_Limit'] = test_df.groupby(level='Symbol')['Limit_Entry_Price'].shift(1)
    test_df['Pending_Buy_Target'] = test_df.groupby(level='Symbol')['Target_Exit_Price'].shift(1)
    test_df['Pending_Buy_Signal'] = test_df.groupby(level='Symbol')['Signal'].shift(1).fillna(False).infer_objects()
    
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    positions = []
    
    for date in tqdm(sim_dates, desc="Simulating Days"):
        try:
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
            
        remaining_positions = []
        
        # 1. Manage Existing Positions
        for pos in positions:
            symbol = pos['Symbol']
            if symbol not in day_data.index:
                remaining_positions.append(pos)
                continue
                
            row = day_data.loc[symbol]
            high = row['High']
            close = row['Close']
            prev_high = row['Prev_High']
            
            exit_price = None
            exit_reason = None
            
            if high >= pos['Target']:
                if row['Open'] > pos['Target']:
                    exit_price = row['Open']
                else:
                    exit_price = pos['Target']
                exit_reason = "Target"
                
            elif close > prev_high:
                exit_price = close
                exit_reason = "Close > PrevHigh"
                
            pos['Days_Held'] += 1
            if exit_price is None and pos['Days_Held'] >= TIME_STOP_DAYS:
                exit_price = close
                exit_reason = "Time Stop"
            
            if exit_price is not None:
                gross_pnl = (exit_price - pos['Entry_Price']) * pos['Shares']
                comm_exit = max(MIN_COMMISSION, pos['Shares'] * COMMISSION_PER_SHARE)
                net_pnl = gross_pnl - comm_exit - pos['Commission_Entry']
                
                cash += (pos['Shares'] * exit_price) - comm_exit
                
                trades.append({
                    'Symbol': symbol,
                    'Entry_Date': pos['Entry_Date'],
                    'Exit_Date': date,
                    'Entry_Price': pos['Entry_Price'],
                    'Exit_Price': exit_price,
                    'Return': (exit_price - pos['Entry_Price']) / pos['Entry_Price'],
                    'PnL': net_pnl,
                    'Reason': exit_reason,
                    'Days_Held': pos['Days_Held'],
                    'Comm_Entry': pos['Commission_Entry'],
                    'Comm_Exit': comm_exit
                })
            else:
                remaining_positions.append(pos)
                
        positions = remaining_positions
        
        # 2. Process New Entries
        if len(positions) < MAX_POSITIONS:
            candidates = day_data[day_data['Pending_Buy_Signal']].copy()
            
            for symbol, row in candidates.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    break
                
                if any(p['Symbol'] == symbol for p in positions):
                    continue
                
                limit_price = row['Pending_Buy_Limit']
                target_price = row['Pending_Buy_Target']
                open_price = row['Open']
                low_price = row['Low']
                
                entry_price = None
                
                if open_price <= limit_price:
                    entry_price = open_price
                elif low_price <= limit_price:
                    entry_price = limit_price
                
                if entry_price:
                    # Calculate total equity for sizing
                    current_portfolio_value = 0
                    for p in positions:
                        s_sym = p['Symbol']
                        if s_sym in day_data.index:
                            current_portfolio_value += p['Shares'] * day_data.loc[s_sym]['Close']
                        else:
                            current_portfolio_value += p['Shares'] * p['Entry_Price']
                    
                    total_equity = cash + current_portfolio_value
                    target_allocation = total_equity * trade_pct_equity
                    
                    if target_allocation > cash:
                        target_allocation = cash
                    
                    shares = int(target_allocation / entry_price)
                    
                    if shares > 0:
                        comm_entry = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)
                        if (shares * entry_price + comm_entry) > cash:
                             shares = int((cash - MIN_COMMISSION) / (entry_price + COMMISSION_PER_SHARE))
                             comm_entry = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)
                        
                        if shares > 0:
                            cash -= (shares * entry_price + comm_entry)
                            positions.append({
                                'Symbol': symbol,
                                'Entry_Price': entry_price,
                                'Shares': shares,
                                'Entry_Date': date,
                                'Target': target_price,
                                'Days_Held': 0,
                                'Commission_Entry': comm_entry
                            })
        
        # Calculate Daily Stats
        pos_value = 0
        for pos in positions:
            sym = pos['Symbol']
            price = pos['Entry_Price']
            if sym in day_data.index:
                price = day_data.loc[sym]['Close']
            pos_value += price * pos['Shares']
            
        total_equity = cash + pos_value
        equity_curve.append({
            'Date': date,
            'Equity': total_equity,
            'Cash': cash,
            'Positions_Value': pos_value,
            'Cash_Util': pos_value / total_equity if total_equity > 0 else 0,
            'Num_Positions': len(positions)
        })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def calculate_cagr(final_equity, years):
    return (final_equity / STARTING_CAPITAL) ** (1 / years) - 1

def main():
    symbols = get_all_symbols()
    if not symbols:
        print("No symbols found.")
        return

    df = download_and_cache_data(symbols)
    df = prepare_data(df)
    
    # Download Benchmark
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    spy, benchmark_ticker = download_spy(start_date, df_universe=df)
    
    results = []
    
    print("\n" + "="*80)
    print(f"STARTING PARAMETER SWEEP: Position Sizes {SIZES_TO_TEST}")
    print("="*80)
    
    script_dir = Path(__file__).resolve().parent

    best_eq_df = None
    best_stats = None
    
    for size in SIZES_TO_TEST:
        eq_df, trades_df = run_simulation(df, size)
        
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
        
        # Correlation with Benchmark (Active Days Only)
        correlation = 0
        if not spy.empty:
             # Use Date as index for alignment
             temp_df = eq_df.set_index('Date')
             
             # Filter for days where we had positions (exclude 100% cash days)
             active_days = temp_df[temp_df['Num_Positions'] > 0].index
             
             strat_ret_series = temp_df.loc[active_days, 'Daily_Ret']
             spy_ret_series = spy['Close'].pct_change()
             
             # Align
             common_idx = strat_ret_series.index.intersection(spy_ret_series.index)
             
             if len(common_idx) > 10:
                 correlation = strat_ret_series.loc[common_idx].corr(spy_ret_series.loc[common_idx])
        
        # Calculate SQN
        sqn = 0
        if not trades_df.empty:
            trade_rets = trades_df['Return']
            if trade_rets.std() > 0:
                sqn = np.sqrt(len(trades_df)) * (trade_rets.mean() / trade_rets.std())
        
        # Calculate Exposure Metrics
        exposure_time_pct = (len(eq_df[eq_df['Num_Positions'] > 0]) / len(eq_df)) * 100
        avg_exposure_pct = eq_df['Cash_Util'].mean() * 100

        stats = {
            "Size %": size * 100,
            "CAGR %": cagr * 100,
            "Total Ret %": total_ret * 100,
            "Sharpe": sharpe,
            "Max DD %": max_dd * 100,
            "Corr SPY": correlation,
            "SQN": sqn,
            "Exposure %": exposure_time_pct,
            "Trades": len(trades_df),
            "Final Eq": final_equity
        }
        results.append(stats)
        
        # Keep the best run (highest Sharpe or user choice - let's default to highest CAGR for plot interest)
        if best_stats is None or cagr > best_stats['CAGR %'] / 100:
            best_stats = stats
            best_eq_df = eq_df
    
    # Print Summary
    res_df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("PARAMETER SWEEP RESULTS")
    print("="*100)
    cols = ["Size %", "CAGR %", "Total Ret %", "Sharpe", "Max DD %", "SQN", "Exposure %", "Corr SPY", "Trades", "Final Eq"]
    print(res_df[cols].to_string(index=False, formatters={
        "Final Eq": "${:,.0f}".format,
        "CAGR %": "{:.2f}".format,
        "Total Ret %": "{:.2f}".format,
        "Sharpe": "{:.2f}".format,
        "Max DD %": "{:.2f}".format,
        "SQN": "{:.2f}".format,
        "Exposure %": "{:.2f}".format,
        "Corr SPY": "{:.2f}".format
    }))
    print("="*100)
    
    # Plotting Best Run
    if best_eq_df is not None:
        best_size = best_stats['Size %']
        print(f"\nPlotting results for best run: {best_size}% Size")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. Equity Curve vs Benchmark
        # Normalize Benchmark to start at same capital
        spy_aligned = spy[spy.index >= best_eq_df.iloc[0]['Date']].copy()
        if not spy_aligned.empty:
            spy_aligned['Norm'] = (spy_aligned['Close'] / spy_aligned.iloc[0]['Close']) * STARTING_CAPITAL
            ax1.plot(spy_aligned.index, spy_aligned['Norm'], label=f"{benchmark_ticker} (Benchmark)", color='gray', alpha=0.6, linestyle='--')
            
        ax1.plot(best_eq_df['Date'], best_eq_df['Equity'], label=f"Strategy ({best_size}%)", color='blue')
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(f"Panic Dip + ATR Strategy (Size: {best_size}%) vs {benchmark_ticker}\nCAGR: {best_stats['CAGR %']:.2f}% | Max DD: {best_stats['Max DD %']:.2f}% | Sharpe: {best_stats['Sharpe']:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Cash Utilization
        ax2.plot(best_eq_df['Date'], best_eq_df['Cash_Util'] * 100, color='orange', label="Cash Util %")
        ax2.set_ylabel("Invested %")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3.fill_between(best_eq_df['Date'], best_eq_df['Drawdown'] * 100, 0, color='red', alpha=0.3, label="Drawdown %")
        ax3.set_ylabel("Drawdown %")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(script_dir / "b_panic_dip_summary.png")
        print(f"Comparison plot saved to {script_dir / 'b_panic_dip_summary.png'}")

if __name__ == "__main__":
    main()
