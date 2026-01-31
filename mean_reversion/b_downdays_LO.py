import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import arcticdb as adb
import dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 100_000
MAX_TRADE_PERCENT_EQUITY = 0.05 # 0.05 is good
HISTORY_FILE = "us_stock_history_10y.csv"
PERIOD_YEARS_DOWNLOAD = "10y"
BACKTEST_YEARS = 5
EMA_WINDOW = 200
DECLINE_DAYS = 4
MULTIPLIERS_TO_TEST = [1.5, 2.0, 2.5, 3.0]

def load_arctic():
    """Load ArcticDB connection using environment variables."""
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
    print("Connecting to ArcticDB to fetch symbol list...")
    ac = load_arctic()
    try:
        lib = ac.get_library("us_equities")
        all_stocks = lib.read("ALL_STOCKS").data
        print(all_stocks.tail())
    except:
        print("Could not read ALL_STOCKS from us_equities.")
        return []
    
    if "Symbol" in all_stocks.columns:
        symbols = sorted(all_stocks["Symbol"].unique().tolist())
    elif "Ticker" in all_stocks.columns:
        symbols = sorted(all_stocks["Ticker"].unique().tolist())
    else:
        symbols = sorted(all_stocks.index.unique().tolist())
    
    print(f"Found {len(symbols)} symbols.")
    return symbols

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

    print(f"Downloading {PERIOD_YEARS_DOWNLOAD} of data for {len(symbols)} symbols. This may take a while...")
    
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

    print("Concatenating data...")
    full_df = pd.concat(all_dfs)
    full_df = full_df.sort_index()
    
    print(f"Saving data to {csv_path}...")
    full_df.to_csv(csv_path)
    
    return full_df

def prepare_data(df):
    """
    Calculate technical indicators required for the strategy.
    """
    print("Calculating indicators...")
    
    df = df.sort_index()
    
    if "Close" not in df.columns or "Open" not in df.columns:
        df.rename(columns={"close": "Close", "open": "Open"}, inplace=True)

    # 1. Calculate 200 EMA
    print("Computing 200 EMA...")
    df['EMA_200'] = df.groupby(level='Symbol')['Close'].transform(
        lambda x: x.ewm(span=EMA_WINDOW, adjust=False).mean()
    )

    # 2. Calculate Daily Returns and Declines
    print("Computing returns and declines...")
    df['Daily_Ret'] = df.groupby(level='Symbol')['Close'].pct_change()
    
    df['Prev_Close'] = df.groupby(level='Symbol')['Close'].shift(1)
    df['Is_Decline'] = df['Close'] < df['Prev_Close']
    
    # 3. Calculate Consecutive Declines
    def calculate_streaks(group):
        condition = group['Is_Decline']
        s = condition
        cumsum = s.cumsum()
        reset = cumsum.where(~s).ffill().fillna(0).astype(int)
        return cumsum - reset

    df['Decline_Streak'] = df.groupby(level='Symbol', group_keys=False).apply(calculate_streaks)

    # 4. Calculate Mean Return of last 4 days
    print("Computing 4-day mean return...")
    df['Mean_4D_Ret'] = df.groupby(level='Symbol')['Daily_Ret'].rolling(window=4).mean().reset_index(level=0, drop=True)

    return df

def simulate_multiplier(df, multiplier):
    """
    Run the backtest loop with a specific Limit Order Multiplier.
    Limit Price = Close * (1 + multiplier * Mean_4D_Ret)
    """
    print(f"\n>>> Simulating with Limit Multiplier: {multiplier}x")
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    # Identify signals
    test_df['Signal'] = (test_df['Close'] > test_df['EMA_200']) & (test_df['Decline_Streak'] == DECLINE_DAYS)
    
    # Logic for Limit Price with Multiplier
    test_df['Limit_Price_Target'] = test_df['Close'] * (1 + (multiplier * test_df['Mean_4D_Ret']))
    
    # Shift signal and target price to the next day (Trade Day)
    test_df['Trade_Day_Signal'] = test_df.groupby(level='Symbol')['Signal'].shift(1).fillna(False).infer_objects()
    test_df['Limit_Order_Price'] = test_df.groupby(level='Symbol')['Limit_Price_Target'].shift(1)
    
    test_df['Date_Col'] = test_df.index.get_level_values('Date')
    test_df['Signal_Date'] = test_df.groupby(level='Symbol')['Date_Col'].shift(1)
    
    current_equity = cash
    
    for date in tqdm(sim_dates, desc=f"Simulating ({multiplier}x)"):
        try:
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
            
        # Identify symbols to trade
        candidates = day_data[day_data['Trade_Day_Signal']].copy()
        
        if candidates.empty:
            equity_curve.append({'Date': date, 'Equity': current_equity, 'Cash': current_equity, 'Positions': 0, 'Exposure': 0})
            continue
        
        # Determine position size
        target_size = current_equity * MAX_TRADE_PERCENT_EQUITY
        potential_cost = len(candidates) * target_size
        
        actual_trade_size = target_size
        if potential_cost > current_equity:
            actual_trade_size = current_equity / len(candidates)
        
        daily_pnl = 0
        active_trades_count = 0
        total_exposure = 0
        
        for symbol, row in candidates.iterrows():
            open_price = row['Open']
            close_price = row['Close']
            low_price = row['Low']
            limit_price = row['Limit_Order_Price']
            
            if pd.isna(open_price) or pd.isna(close_price) or pd.isna(limit_price) or open_price == 0:
                continue
            
            # Execution Logic for Limit Order
            entry_price = None
            
            # Case 1: Gap Down
            if open_price <= limit_price:
                entry_price = open_price
            # Case 2: Intraday Fill
            elif low_price <= limit_price:
                entry_price = limit_price
            else:
                continue
                
            # Calculate return (Sell at Close)
            ret = (close_price - entry_price) / entry_price
            pnl = actual_trade_size * ret
            
            daily_pnl += pnl
            active_trades_count += 1
            total_exposure += actual_trade_size
            
            trades.append({
                'Date': date,
                'Symbol': symbol,
                'Entry_Price': entry_price,
                'Exit_Price': close_price,
                'Return': ret,
                'PnL': pnl,
            })
            
        current_equity += daily_pnl
        equity_curve.append({'Date': date, 'Equity': current_equity, 'Cash': current_equity, 'Positions': active_trades_count, 'Exposure': total_exposure})

    # Calculate Stats
    if not equity_curve:
        return None, None, None

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    
    final_equity = equity_df.iloc[-1]['Equity']
    total_return = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    equity_df['Daily_Ret'] = equity_df['Equity'].pct_change()
    daily_mean = equity_df['Daily_Ret'].mean()
    daily_std = equity_df['Daily_Ret'].std()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    
    if not trades_df.empty:
        win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
        avg_trade_ret = trades_df['Return'].mean() * 100
        trade_count = len(trades_df)
        gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        win_rate = 0
        avg_trade_ret = 0
        trade_count = 0
        pf = 0
        
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_dd = equity_df['Drawdown'].min() * 100
    
    stats = {
        "Multiplier": multiplier,
        "Final Equity": final_equity,
        "Total Return %": total_return,
        "Sharpe": sharpe,
        "Trades": trade_count,
        "Win Rate %": win_rate,
        "Avg Trade %": avg_trade_ret,
        "Max DD %": max_dd,
        "Profit Factor": pf
    }
    
    return stats, equity_df, trades_df

def main():
    symbols = get_all_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        return

    df = download_and_cache_data(symbols)
    df = prepare_data(df)
    
    results = []
    
    print("\n" + "="*50)
    print(f"STARTING COMPARISON: 4D DECLINE with LIMIT MULTIPLIERS")
    print(f"Multipliers: {MULTIPLIERS_TO_TEST}")
    print("="*50)
    
    script_dir = Path(__file__).resolve().parent

    # Prepare Plot
    plt.figure(figsize=(12, 8))

    for mult in MULTIPLIERS_TO_TEST:
        stats, eq_df, tr_df = simulate_multiplier(df, mult)
        if stats:
            results.append(stats)
            # Save individual runs - COMMENTED OUT
            # eq_df.to_csv(script_dir / f"b_downdays_LO_m{mult}_equity.csv", index=False)
            # tr_df.to_csv(script_dir / f"b_downdays_LO_m{mult}_trades.csv", index=False)
            
            # Plot Equity Curve
            plt.plot(eq_df['Date'], eq_df['Equity'], label=f"Multiplier {mult}x (Total Ret: {stats['Total Return %']:.0f}%)")
    
    # Finalize Plot
    plt.title("Equity Curve Comparison - Limit Order Multipliers")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(script_dir / "b_downdays_LO_comparison.png")
    print(f"\nComparison plot saved to {script_dir / 'b_downdays_LO_comparison.png'}")

    # Print Comparison Table
    if results:
        res_df = pd.DataFrame(results)
        # Format for display
        res_df = res_df.round(2)
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        # Reorder columns
        cols = ["Multiplier", "Final Equity", "Total Return %", "Sharpe", "Max DD %", "Win Rate %", "Avg Trade %", "Trades", "Profit Factor"]
        print(res_df[cols].to_string(index=False))
        print("="*80)
        
        # Save summary
        # res_df.to_csv(script_dir / "b_downdays_LO_comparison.csv", index=False)
        print(f"Summary saved to {script_dir}")

if __name__ == "__main__":
    main()
