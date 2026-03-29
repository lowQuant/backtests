import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 100_000
HISTORY_FILE = "us_stock_history_10y.csv"
PERIOD_YEARS_DOWNLOAD = "10y"
BACKTEST_YEARS = 10 

# Strategy Parameters
MIN_DOLLAR_VOLUME = 100_000_000 # $100m
DOLLAR_VOLUME_WINDOW = 50
HV_MIN = 0.40 # > 40% (High Volatility, complementary to LTlowvol's 10-40%)
HV_MAX = 5.00 # No Upper Limit
HV_WINDOW = 20
SMA_WINDOW = 100 # Faster trend filter for volatile stocks
BREAKOUT_WINDOW = 20 # 20-day High Breakout
ATR_WINDOW = 20 # Faster ATR for volatile stocks

# Risk Management
RISK_PER_TRADE = 0.01 # 1% Risk (Lower due to higher volatility)
MAX_POS_SIZE = 0.05 # 5% Max Position (More diversification)
STOP_LOSS_ATR_MULT = 3.0 # Wide stop for high vol
TRAILING_STOP_PCT = 0.25 # 25% Trailing Stop (Loose to allow runners)

COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0

def get_all_symbols():
    """Fetch symbols. Using a default list if no file or DB."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Check mean_reversion folder first (user specified location)
    mr_csv_path = project_root / "mean_reversion" / HISTORY_FILE
    root_csv_path = project_root / HISTORY_FILE
    
    csv_path = None
    if mr_csv_path.exists():
        csv_path = mr_csv_path
    elif root_csv_path.exists():
        csv_path = root_csv_path
    
    if csv_path:
        print(f"Local history file found at {csv_path}. Using symbols from there.")
        try:
            df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
            symbols = df.index.get_level_values(1).unique().tolist()
            return symbols
        except Exception as e:
            print(f"Error reading local file: {e}")
            
    # Fallback default list
    print("No history file found. Using default list.")
    return ["SPY", "QQQ", "IWM", "TSLA", "NVDA", "AMD", "ENPH", "SEDG", "COIN", "MARA", "RIOT", "PLTR", "DKNG", "ROKU", "SQ", "SHOP", "NET", "DDOG"]

def download_spy(start_date):
    """Download SPY data for filter and benchmark."""
    print("Downloading SPY data...")
    try:
        spy = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                try:
                    if "SPY" in spy.columns.get_level_values(0):
                        spy = spy.xs("SPY", axis=1, level=0)
                    elif "SPY" in spy.columns.get_level_values(1):
                        spy = spy.xs("SPY", axis=1, level=1)
                except Exception as e:
                    print(f"Error extracting SPY data from MultiIndex: {e}")
            
            if "Close" in spy.columns:
                return spy
    except Exception as e:
        print(f"Error downloading SPY: {e}")
    return pd.DataFrame()

def download_and_cache_data(symbols):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    mr_csv_path = project_root / "mean_reversion" / HISTORY_FILE
    root_csv_path = project_root / HISTORY_FILE
    
    csv_path = root_csv_path
    if mr_csv_path.exists():
        csv_path = mr_csv_path
    elif root_csv_path.exists():
        csv_path = root_csv_path
    
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
            if data.empty: continue
            
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
    print("Calculating indicators...")
    df = df.sort_index()
    
    if "Close" not in df.columns or "Open" not in df.columns:
        df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)

    g = df.groupby(level='Symbol')
    
    # 1. Dollar Volume
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['Avg_Dollar_Volume'] = g['Dollar_Volume'].transform(lambda x: x.rolling(window=DOLLAR_VOLUME_WINDOW).mean())
    
    # 2. Historic Volatility
    df['Log_Ret'] = g['Close'].transform(lambda x: np.log(x / x.shift(1)))
    df['HV'] = g['Log_Ret'].transform(lambda x: x.rolling(window=HV_WINDOW).std() * np.sqrt(252))
    
    # 3. SMA (Trend Filter)
    df['SMA_Trend'] = g['Close'].transform(lambda x: x.rolling(window=SMA_WINDOW).mean())
    
    # 4. Momentum / Breakout Indicator
    # Donchian Channel High (Max of past N days, excluding today)
    # Actually, we want to check if TODAY's Close > Max(Close of previous N days)
    # So we calculate rolling max of previous N days.
    df['Rolling_Max_Close'] = g['Close'].transform(lambda x: x.shift(1).rolling(window=BREAKOUT_WINDOW).max())
    
    # ROC 20 (Rate of Change) for Ranking
    df['ROC_20'] = g['Close'].transform(lambda x: x.pct_change(periods=20))
    
    # 5. ATR
    h = df['High']
    l = df['Low']
    pc = g['Close'].shift(1)
    
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['ATR'] = g.apply(lambda x: tr.loc[x.index].rolling(window=ATR_WINDOW).mean()).reset_index(level=0, drop=True)
    
    return df

def run_simulation_high_iv(df, spy_df):
    print("Pre-calculating signals...")
    
    spy_df = spy_df.copy()
    spy_df['SPY_SMA'] = spy_df['Close'].rolling(window=200).mean()
    spy_df['SPY_Above_SMA'] = spy_df['Close'] > spy_df['SPY_SMA']
    
    # Reindex SPY signal to match stock data dates
    # Assuming df is sorted by Date, we can map
    df = df.reset_index().set_index('Date').sort_index()
    
    spy_series = spy_df['SPY_Above_SMA']
    # Efficient mapping
    df['SPY_Filter'] = df.index.map(spy_series).fillna(False)
    
    # Apply Filters
    # 1. High Volatility (> 40%)
    # 2. Liquidity (> $50M)
    # 3. Trend (Close > SMA 100)
    # 4. Market Regime (SPY > SMA 200)
    # 5. Breakout (Close > 20d High)
    
    mask = (
        (df['Avg_Dollar_Volume'] > MIN_DOLLAR_VOLUME) &
        (df['HV'] >= HV_MIN) &
        (df['HV'] <= HV_MAX) &
        (df['Close'] > df['SMA_Trend']) &
        (df['SPY_Filter']) &
        (df['Close'] > df['Rolling_Max_Close']) # Breakout Signal
    )
    
    df['Signal'] = mask
    
    # Prepare for iteration
    # Entry Signal for Day T is Signal from Day T-1 (We enter on Open of T)
    df = df.reset_index().set_index(['Symbol', 'Date']).sort_index()
    
    # Shift Signal to create Entry Trigger
    df['Entry_Signal'] = df.groupby(level='Symbol')['Signal'].shift(1).fillna(False)
    
    # Shift other metrics needed for entry logic (computed at T-1)
    df['Entry_ATR'] = df.groupby(level='Symbol')['ATR'].shift(1)
    df['Entry_ROC'] = df.groupby(level='Symbol')['ROC_20'].shift(1)
    
    print(f"\n>>> Simulating Strategy: High Volatility Breakout")
    
    df = df.reset_index().set_index('Date').sort_index()
    dates = df.index.unique()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    sim_dates = dates[dates >= start_date]
    
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    positions = [] 
    
    for date in tqdm(sim_dates, desc="Simulating"):
        try:
            day_data = df.loc[date]
        except KeyError:
            continue
            
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T
            
        # Create price lookup
        # Some rows might have duplicates if data error, handle gracefully
        day_data = day_data[~day_data.index.duplicated(keep='first')] # Symbol is column now due to reset_index earlier? 
        # Wait, set_index('Date') leaves Symbol as column.
        
        # Check if Symbol is in columns
        if 'Symbol' not in day_data.columns:
            # If it's a Series, name might be Symbol? No, reset_index made it a column.
            # If single row dataframe, it works.
            pass

        current_prices = day_data.set_index('Symbol')[['Open', 'High', 'Low', 'Close']].to_dict('index')
        
        remaining_positions = []
        
        # 1. Check Exits
        for pos in positions:
            sym = pos['Symbol']
            if sym not in current_prices:
                remaining_positions.append(pos)
                continue
                
            prices = current_prices[sym]
            open_p = prices['Open']
            high_p = prices['High']
            low_p = prices['Low']
            close_p = prices['Close']
            
            # Update High Water Mark
            if high_p > pos['High_Water_Mark']:
                pos['High_Water_Mark'] = high_p
            
            # Trailing Stop
            trailing_stop = pos['High_Water_Mark'] * (1.0 - TRAILING_STOP_PCT)
            
            # Stop Loss (Fixed initial stop vs Trailing)
            # Use the higher of the two
            stop_price = max(pos['Fixed_Stop_Price'], trailing_stop)
            
            exit_price = None
            exit_reason = None
            
            # Check Stop
            if low_p <= stop_price:
                exit_price = min(open_p, stop_price) # Gap handling
                exit_reason = "Stop"
            
            if exit_price is not None:
                shares = pos['Shares']
                gross_pnl = (exit_price - pos['Entry_Price']) * shares
                comm_exit = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)
                net_pnl = gross_pnl - comm_exit - pos['Commission_Entry']
                
                cash += (exit_price * shares) - comm_exit
                
                trades.append({
                    'Symbol': sym,
                    'Entry_Date': pos['Entry_Date'],
                    'Exit_Date': date,
                    'Entry_Price': pos['Entry_Price'],
                    'Exit_Price': exit_price,
                    'Return': (exit_price - pos['Entry_Price']) / pos['Entry_Price'],
                    'PnL': net_pnl,
                    'Reason': exit_reason
                })
            else:
                remaining_positions.append(pos)
        
        positions = remaining_positions
        
        # 2. Check Entries
        # Candidates where Entry_Signal is True
        candidates = day_data[day_data['Entry_Signal']].copy()
        
        # Remove held positions
        held_symbols = {p['Symbol'] for p in positions}
        candidates = candidates[~candidates['Symbol'].isin(held_symbols)]
        
        if not candidates.empty:
            # Rank by Highest Momentum (ROC 20)
            candidates = candidates.sort_values('Entry_ROC', ascending=False)
            
            for _, row in candidates.iterrows():
                sym = row['Symbol']
                entry_price = row['Open']
                atr = row['Entry_ATR']
                
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Position Sizing
                # Risk = 1% Equity
                # Stop Distance = 3 * ATR
                
                current_equity = cash + sum([p['Shares'] * current_prices.get(p['Symbol'], {'Close': p['Entry_Price']})['Close'] for p in positions])
                risk_amount = current_equity * RISK_PER_TRADE
                stop_distance = STOP_LOSS_ATR_MULT * atr
                
                # Shares based on risk
                if stop_distance == 0: continue
                shares = int(risk_amount / stop_distance)
                
                # Cap Size: 5% of Equity
                max_size_val = current_equity * MAX_POS_SIZE
                if shares * entry_price > max_size_val:
                    shares = int(max_size_val / entry_price)
                
                # Check Cash
                comm_est = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)
                cost = shares * entry_price + comm_est
                
                if cost > cash:
                    shares = int((cash - MIN_COMMISSION) / (entry_price + COMMISSION_PER_SHARE))
                    comm_est = max(MIN_COMMISSION, shares * COMMISSION_PER_SHARE)
                    cost = shares * entry_price + comm_est
                
                if shares > 0:
                    cash -= cost
                    fixed_stop = entry_price - stop_distance
                    
                    positions.append({
                        'Symbol': sym,
                        'Entry_Date': date,
                        'Entry_Price': entry_price,
                        'Shares': shares,
                        'Fixed_Stop_Price': fixed_stop,
                        'High_Water_Mark': entry_price,
                        'Commission_Entry': comm_est
                    })
                    
        # 3. Record Equity
        current_val = 0
        for p in positions:
            sym = p['Symbol']
            if sym in current_prices:
                current_val += p['Shares'] * current_prices[sym]['Close']
            else:
                current_val += p['Shares'] * p['Entry_Price']
                
        total_equity = cash + current_val
        equity_curve.append({
            'Date': date,
            'Equity': total_equity,
            'Cash': cash,
            'Drawdown': 0.0
        })
        
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def main():
    symbols = get_all_symbols()
    if not symbols:
        return

    df = download_and_cache_data(symbols)
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS + 1)
    spy = download_spy(start_date)
    
    df = prepare_data(df)
    
    eq_df, trades_df = run_simulation_high_iv(df, spy)
    
    if eq_df.empty:
        print("No results generated.")
        return

    # Metrics
    final_equity = eq_df.iloc[-1]['Equity']
    total_ret = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL
    cagr = (final_equity / STARTING_CAPITAL) ** (1 / BACKTEST_YEARS) - 1
    
    eq_df['Daily_Log_Ret'] = np.log(eq_df['Equity'] / eq_df['Equity'].shift(1))
    eq_df['Daily_Ret'] = eq_df['Equity'].pct_change()
    
    sharpe = (eq_df['Daily_Ret'].mean() / eq_df['Daily_Ret'].std()) * np.sqrt(252) if eq_df['Daily_Ret'].std() > 0 else 0
    
    pos_log_rets = eq_df[eq_df['Daily_Log_Ret'] > 0]['Daily_Log_Ret'].sum()
    neg_log_rets = abs(eq_df[eq_df['Daily_Log_Ret'] < 0]['Daily_Log_Ret'].sum())
    profit_factor = pos_log_rets / neg_log_rets if neg_log_rets > 0 else float('inf')
    
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']
    max_dd = eq_df['Drawdown'].min()
    
    win_rate = 0
    avg_win_pct = 0
    avg_loss_pct = 0
    
    if not trades_df.empty:
        wins = trades_df[trades_df['PnL'] > 0]
        losses = trades_df[trades_df['PnL'] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100
        avg_win_pct = wins['Return'].mean() * 100 if not wins.empty else 0
        avg_loss_pct = losses['Return'].mean() * 100 if not losses.empty else 0
        
    print("\n" + "="*60)
    print("RESULTS: High Volatility Momentum (LT_highIV)")
    print("="*60)
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Profit Factor:      {profit_factor:.2f} (Daily Log Returns)")
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Avg Win:            {avg_win_pct:.2f}%")
    print(f"Avg Loss:           {avg_loss_pct:.2f}%")
    print(f"Total Return:       {total_ret*100:.2f}%")
    print(f"Final Equity:       ${final_equity:,.2f}")
    print(f"Total Trades:       {len(trades_df)}")
    print("="*60)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(eq_df['Date'], eq_df['Equity'], label="Equity", color='purple')
    ax1.set_title("Total Equity - High Vol Momentum")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.fill_between(eq_df['Date'], eq_df['Drawdown'] * 100, 0, color='orange', alpha=0.3, label="Drawdown %")
    ax2.set_title("Max Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(__file__).parent / "LT_highIV_results.png"
    plt.savefig(plot_path)
    print(f"Chart saved to {plot_path}")

if __name__ == "__main__":
    main()
