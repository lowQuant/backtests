import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import arcticdb as adb
import dotenv
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 100_000
MAX_TRADE_PERCENT_EQUITY = 0.05
HISTORY_FILE = "us_stock_history_10y.csv"
PERIOD_YEARS_DOWNLOAD = "10y"
BACKTEST_YEARS = 5
EMA_WINDOW = 200
DECLINE_DAYS = 4

def load_arctic():
    """Load ArcticDB connection using environment variables."""
    dotenv.load_dotenv()
    bucket_name = os.getenv("BUCKET_NAME")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not bucket_name or not aws_access_key_id or not aws_secret_access_key or not aws_region:
        # Try the '2' suffix variables if the main ones aren't set (based on user's other script)
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
    print("Connecting to ArcticDB to fetch symbol list...")
    ac = load_arctic()
    # Try 'us_equities' first, as seen in previous scripts
    try:
        lib = ac.get_library("us_equities")
        all_stocks = lib.read("ALL_STOCKS").data
    except:
        # Fallback or error handling if library doesn't exist
        print("Could not read ALL_STOCKS from us_equities.")
        return []
    
    if "Symbol" in all_stocks.columns:
        symbols = sorted(all_stocks["Symbol"].unique().tolist())
    elif "Ticker" in all_stocks.columns:
        symbols = sorted(all_stocks["Ticker"].unique().tolist())
    else:
        symbols = sorted(all_stocks.index.unique().tolist()) # Sometimes index is symbol
    
    print(f"Found {len(symbols)} symbols.")
    return symbols

def download_and_cache_data(symbols):
    """
    Download 10y historical data for all symbols and save to CSV.
    If CSV exists, load from it instead.
    """
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

    print(f"Downloading {PERIOD_YEARS_DOWNLOAD} of data for {len(symbols)} symbols. This may take a while...")
    
    # Chunking symbols to avoid overwhelming yfinance or memory issues
    chunk_size = 500
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    all_dfs = []

    for chunk in tqdm(chunks, desc="Downloading chunks"):
        try:
            # auto_adjust=False to keep Close and Adj Close (though we mainly need Close/Open)
            # Using 'auto_adjust=True' simplifies Close to be split/div adjusted which is good for backtesting
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
            
            # Stack to get proper format: Index=[Date, Symbol], Columns=[Open, Close, ...]
            # yfinance multi-index columns are (Price, Ticker) or (Ticker, Price) depending on input
            # If group_by='ticker', columns are (Ticker, Price)
            
            # Check structure
            if isinstance(data.columns, pd.MultiIndex):
                # data is (Date) x (Ticker, Price)
                # We want to stack to get (Date, Ticker) x (Price)
                stacked = data.stack(level=0)
                
                # If the stack resulted in Price as index and Ticker as columns (rare but possible), handle it.
                # With group_by='ticker', level 0 is Ticker.
                # Actually yf.download with group_by='ticker' returns columns: (Ticker, OHLCV)
                # So stacking level 0 puts Ticker into index.
                
                stacked.index.names = ['Date', 'Symbol']
                all_dfs.append(stacked)
            else:
                # Single symbol case
                # Add symbol column
                data['Symbol'] = chunk[0]
                data = data.set_index('Symbol', append=True)
                all_dfs.append(data)
                
        except Exception as e:
            print(f"Error downloading chunk: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data downloaded.")

    print("Concatenating data...")
    full_df = pd.concat(all_dfs)
    full_df = full_df.sort_index()
    
    print(f"Saving data to {csv_path}...")
    full_df.to_csv(csv_path)
    
    return full_df

def prepare_data(df):
    """
    Calculate technical indicators required for the strategy.
    - 200 EMA
    - Consecutive Decline Count
    """
    print("Calculating indicators...")
    
    # Ensure we work with a copy and proper sorting
    df = df.sort_index()
    
    # We need Open and Close
    if "Close" not in df.columns or "Open" not in df.columns:
        # Handle case where columns might be lower case
        df.rename(columns={"close": "Close", "open": "Open"}, inplace=True)

    # Group by Symbol for vector operations
    # Reset index to operations easier then set back? 
    # Or just use groupby transform/apply
    
    # 1. Calculate 200 EMA
    # Using pandas ewm for EMA. span=200 roughly equates to 200 day EMA
    # talib.EMA is precise, but pandas is faster to implement without extra deps if not needed.
    # User mentioned >200 EMA.
    print("Computing 200 EMA...")
    # span=N corresponds to alpha=2/(N+1). 
    df['EMA_200'] = df.groupby(level='Symbol')['Close'].transform(
        lambda x: x.ewm(span=EMA_WINDOW, adjust=False).mean()
    )

    # 2. Calculate Daily Returns (Log or Simple) to check for decline
    # Decline means Close < Prev Close
    print("Computing declines...")
    df['Daily_Ret'] = df.groupby(level='Symbol')['Close'].pct_change()
    df['Log_Ret'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    
    df['Prev_Close'] = df.groupby(level='Symbol')['Close'].shift(1)
    df['Is_Decline'] = df['Close'] < df['Prev_Close']
    
    # 3. Calculate Consecutive Declines
    # This is tricky to vectorize efficiently over a MultiIndex.
    # We can identify streaks by comparing groups of (Decline != Prev_Decline) cumsum
    
    def calculate_streaks(group):
        # boolean series
        condition = group['Is_Decline']
        # cumsum resets every time condition is false
        # Reference: https://stackoverflow.com/questions/21079716/pandas-groupby-consecutive-values
        # Logic: (Condition).cumsum() - (Condition).cumsum().where(~Condition).ffill().fillna(0)
        # But simpler:
        # y = x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        # where x is the boolean series.
        # However, x needs to be False for non-declines.
        
        c = condition.astype(int)
        # Reset whenever 0
        # Create groups where values change or are 0
        # We only want consecutive 1s
        # We can use the logic: Streak = CumSum - CumSum_at_last_zero
        
        # Alternative loop-free approach for streaks:
        # s = condition
        # s.cumsum() - s.cumsum().where(~s).ffill().fillna(0)
        # This counts consecutive Trues
        
        s = condition
        cumsum = s.cumsum()
        reset = cumsum.where(~s).ffill().fillna(0).astype(int)
        return cumsum - reset

    # Apply streak calculation
    # Using transform is usually faster than apply for uniform return shape
    df['Decline_Streak'] = df.groupby(level='Symbol', group_keys=False).apply(calculate_streaks)

    return df

def run_backtest(df):
    """
    Run the backtest loop.
    """
    print("Running backtest...")
    
    # Filter for the last 5 years
    # Assume df index level 0 is Date
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    
    # Slice dataframe
    # We need a bit of history before start_date for signals (though indicators are already calc'd)
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    # Setup simulation
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    
    # Get unique dates in the test period
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    # Iterate day by day
    # On Day T, we see signals from Day T-1 (Close).
    # We execute Buy On Day T Open.
    # We execute Sell On Day T Close.
    # This is a daily rebalance logic basically.
    
    # To speed up, we can identify all potential trade days/symbols first
    
    # Criteria:
    # 1. Close > 200 EMA
    # 2. Decline Streak >= 4
    
    # Identify signals
    # Signal is generated at Close of Day T. Trade happens on Day T+1.
    # Shift signal by 1 day to align with Trade Day.
    
    test_df['Signal'] = (test_df['Close'] > test_df['EMA_200']) & (test_df['Decline_Streak'] == DECLINE_DAYS)
    
    # We want to trade ON the day AFTER the signal.
    # So if Signal is True on 2023-01-01, we trade on 2023-01-02.
    # Let's shift the signal forward by 1 for each symbol.
    test_df['Trade_Day_Signal'] = test_df.groupby(level='Symbol')['Signal'].shift(1).fillna(False).infer_objects(copy=False)
    
    # Also capture the Signal Date (the previous day) for reporting
    # We can't easily just take "Date - 1" because of weekends/holidays.
    # Best to shift the Date column itself.
    # Create a temporary column for Date to shift it
    test_df['Date_Col'] = test_df.index.get_level_values('Date')
    test_df['Signal_Date'] = test_df.groupby(level='Symbol')['Date_Col'].shift(1)

    # Now we just iterate through days where there is at least one trade
    trade_days = test_df[test_df['Trade_Day_Signal']].index.get_level_values('Date').unique().sort_values()
    
    # To reconstruct the equity curve properly, we should iterate all days or at least fill forward
    # But since trades are intraday (Open to Close), cash is constant between trades unless we track fees/slippage.
    # We will iterate all business days to record daily equity.
    
    current_equity = cash
    
    for date in tqdm(sim_dates, desc="Simulating days"):
        # Get data for this date
        try:
            # loc lookup might be slow in loop, but necessary for day-by-day processing
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
            
        # Identify symbols to trade
        candidates = day_data[day_data['Trade_Day_Signal']]
        
        if candidates.empty:
            equity_curve.append({'Date': date, 'Equity': current_equity, 'Cash': current_equity, 'Positions': 0, 'Exposure': 0})
            continue
        
        # Execute Trades
        # Position Sizing: Cap at 5% of equity, else equal weight
        # If n < 20: each gets 5% (max)
        # If n >= 20: each gets 1/n (equal weight, using 100% cash)
        
        max_size = current_equity * MAX_TRADE_PERCENT_EQUITY
        equal_size = current_equity / len(candidates)
        
        actual_trade_size = min(max_size, equal_size)
        
        total_exposure = len(candidates) * actual_trade_size
        
        # We buy at Open, Sell at Close
        # PnL = (Close - Open) / Open * Trade_Size
        
        daily_pnl = 0
        
        for symbol, row in candidates.iterrows():
            open_price = row['Open']
            close_price = row['Close']
            
            if pd.isna(open_price) or pd.isna(close_price) or open_price == 0:
                continue
                
            # Calculate return
            ret = (close_price - open_price) / open_price
            pnl = actual_trade_size * ret
            
            daily_pnl += pnl
            
            trades.append({
                'Signal_Date': row['Signal_Date'],
                'Date': date,
                'Symbol': symbol,
                'Entry_Price': open_price,
                'Exit_Price': close_price,
                'Return': ret,
                'PnL': pnl,
                'Size': actual_trade_size
            })
            
        current_equity += daily_pnl
        equity_curve.append({'Date': date, 'Equity': current_equity, 'Cash': current_equity, 'Positions': len(candidates), 'Exposure': total_exposure})

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def create_debug_trace(df, trades_df):
    """
    Create a detailed debug trace for one example symbol showing:
    - The 4 consecutive decline days
    - The signal day
    - The trade execution day
    - All relevant prices and calculations
    """
    import json
    
    if trades_df.empty:
        return {"error": "No trades to debug"}
    
    # Pick the first trade that has a valid signal date
    example_trade = trades_df[trades_df['Signal_Date'].notna()].iloc[0]
    symbol = example_trade['Symbol']
    signal_date = pd.Timestamp(example_trade['Signal_Date'])
    trade_date = pd.Timestamp(example_trade['Date'])
    
    # Get data for this symbol around the signal
    symbol_df = df.xs(symbol, level='Symbol').sort_index().copy()
    
    # Calculate signal for this symbol (same logic as in run_backtest)
    symbol_df['Signal'] = (symbol_df['Close'] > symbol_df['EMA_200']) & (symbol_df['Decline_Streak'] == DECLINE_DAYS)
    
    # Find the signal date index
    signal_idx = symbol_df.index.get_loc(signal_date)
    
    # Get 6 days before signal and 2 days after (to show context)
    start_idx = max(0, signal_idx - 6)
    end_idx = min(len(symbol_df), signal_idx + 3)
    
    context_df = symbol_df.iloc[start_idx:end_idx].copy()
    
    # Create structured output
    debug_info = {
        "symbol": symbol,
        "signal_date": str(signal_date.date()),
        "trade_date": str(trade_date.date()),
        "trade_details": {
            "entry_price": float(example_trade['Entry_Price']),
            "exit_price": float(example_trade['Exit_Price']),
            "return_pct": float(example_trade['Return'] * 100),
            "pnl": float(example_trade['PnL']),
            "position_size": float(example_trade['Size'])
        },
        "price_history": []
    }
    
    # Add each day's details
    for date, row in context_df.iterrows():
        day_info = {
            "date": str(date.date()),
            "open": float(row['Open']) if pd.notna(row['Open']) else None,
            "high": float(row['High']) if pd.notna(row['High']) else None,
            "low": float(row['Low']) if pd.notna(row['Low']) else None,
            "close": float(row['Close']) if pd.notna(row['Close']) else None,
            "ema_200": float(row['EMA_200']) if pd.notna(row['EMA_200']) else None,
            "prev_close": float(row['Prev_Close']) if pd.notna(row['Prev_Close']) else None,
            "is_decline": bool(row['Is_Decline']) if pd.notna(row['Is_Decline']) else None,
            "decline_streak": int(row['Decline_Streak']) if pd.notna(row['Decline_Streak']) else None,
            "signal": bool(row['Signal']) if pd.notna(row['Signal']) else None,
            "is_signal_day": date == signal_date,
            "is_trade_day": date == trade_date
        }
        
        # Add annotations
        if date == signal_date:
            day_info["note"] = "SIGNAL GENERATED: 4 consecutive declines & Close > EMA_200"
        elif date == trade_date:
            day_info["note"] = "TRADE EXECUTED: Buy at Open, Sell at Close"
        
        debug_info["price_history"].append(day_info)
    
    return debug_info

def main():
    # 1. Get Symbols
    symbols = get_all_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        return

    # 2. Data Management
    df = download_and_cache_data(symbols)
    
    # 3. Prepare Data
    df = prepare_data(df)
    
    # 4. Run Backtest
    equity_df, trades_df = run_backtest(df)
    
    # 5. Results
    if not equity_df.empty:
        final_equity = equity_df.iloc[-1]['Equity']
        total_return = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        
        # Calculate additional metrics
        equity_df['Daily_Ret'] = equity_df['Equity'].pct_change()
        daily_mean = equity_df['Daily_Ret'].mean()
        daily_std = equity_df['Daily_Ret'].std()
        sharpe_ratio = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Additional Stats
        win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
        avg_trade_return = trades_df['Return'].mean() * 100
        
        # Max Drawdown
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        max_drawdown = equity_df['Drawdown'].min() * 100

        print("-" * 40)
        print(f"Backtest Completed")
        print(f"Starting Capital: ${STARTING_CAPITAL:,.2f}")
        print(f"Final Equity:     ${final_equity:,.2f}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"Total Trades:     {len(trades_df)}")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Avg Trade Ret:    {avg_trade_return:.2f}%")
        print(f"Max Drawdown:     {max_drawdown:.2f}%")
        print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print("-" * 40)
        
        # Save results
        script_dir = Path(__file__).resolve().parent
        equity_df.to_csv(script_dir / "backtest_4d_equity.csv", index=False)
        trades_df.to_csv(script_dir / "backtest_4d_trades.csv", index=False)
        print(f"Results saved to {script_dir}")
        
        # Create debug trace
        print("\nCreating debug trace for example trade...")
        debug_trace = create_debug_trace(df, trades_df)
        debug_path = script_dir / "backtest_debug_trace.json"
        with open(debug_path, 'w') as f:
            import json
            json.dump(debug_trace, f, indent=2)
        print(f"Debug trace saved to {debug_path}")
        print("\n" + "="*60)
        print(f"DEBUG TRACE FOR SYMBOL: {debug_trace['symbol']}")
        print("="*60)
        import json
        print(json.dumps(debug_trace, indent=2))
        print("="*60)

        # Plotting
        try:
            import matplotlib.pyplot as plt
            
            # Calculate Cash Utilization
            # Utilization = Exposure / Equity
            # We stored 'Exposure' in equity_curve
            equity_df['Utilization'] = equity_df['Exposure'] / equity_df['Equity'] * 100

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Equity Curve
            ax1.plot(equity_df['Date'], equity_df['Equity'], label='Equity', color='blue')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Capital ($)')
            ax1.grid(True, which='both', linestyle='--', alpha=0.6)
            ax1.legend()

            # Cash Utilization
            ax2.plot(equity_df['Date'], equity_df['Utilization'], label='Cash Utilization %', color='orange', linewidth=1)
            ax2.fill_between(equity_df['Date'], equity_df['Utilization'], alpha=0.3, color='orange')
            ax2.set_title('Cash Utilization %')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_xlabel('Date')
            ax2.set_ylim(0, 105) # Cap at just over 100%
            ax2.grid(True, which='both', linestyle='--', alpha=0.6)

            plt.tight_layout()
            plot_path = script_dir / "backtest_4d_equity.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            
    else:
        print("No trades generated.")

if __name__ == "__main__":
    main()
