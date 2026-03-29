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
HV_MIN = 0.10 # 10%
HV_MAX = 0.40 # 40%
HV_WINDOW = 20 # 20 days for Historic Volatility
SMA_WINDOW = 200
RSI_WINDOW = 4
ATR_WINDOW = 40

# Risk Management
RISK_PER_TRADE = 0.02 # 2% of equity
MAX_POS_SIZE = 0.10 # 10% of equity
STOP_LOSS_ATR_MULT = 1.5
TRAILING_STOP_PCT = 0.20 # 20%

COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0

def get_all_symbols():
    """Fetch symbols. Using a default list if no file or DB."""
    # Try to find existing history file to get symbols
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
            
    # Fallback default list if no file found (User should ideally provide the file or it will download these)
    print("No history file found. Using default list.")
    return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "JNJ", "V", "PG", "MA", "HD", "UNH", "BAC"]

def download_spy(start_date):
    """Download SPY data for filter and benchmark."""
    print("Downloading SPY data...")
    try:
        spy = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)
        if not spy.empty:
            # Handle MultiIndex (Ticker, Price)
            if isinstance(spy.columns, pd.MultiIndex):
                try:
                    # If Ticker is level 0, select SPY
                    if "SPY" in spy.columns.get_level_values(0):
                        spy = spy.xs("SPY", axis=1, level=0)
                    # Fallback if structure is reversed or different
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
    
    # Check mean_reversion folder first (user specified location)
    mr_csv_path = project_root / "mean_reversion" / HISTORY_FILE
    root_csv_path = project_root / HISTORY_FILE
    
    csv_path = root_csv_path # Default save location if neither exists
    
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
    
    # 1. Dollar Volume (Close * Volume)
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['Avg_Dollar_Volume'] = g['Dollar_Volume'].transform(lambda x: x.rolling(window=DOLLAR_VOLUME_WINDOW).mean())
    
    # 2. Historic Volatility (Annualized Std Dev of Log Returns)
    # Using simple returns approximation or log returns
    df['Log_Ret'] = g['Close'].transform(lambda x: np.log(x / x.shift(1)))
    df['HV'] = g['Log_Ret'].transform(lambda x: x.rolling(window=HV_WINDOW).std() * np.sqrt(252))
    
    # 3. SMA 200
    df['SMA_200'] = g['Close'].transform(lambda x: x.rolling(window=SMA_WINDOW).mean())
    
    # 4. RSI 4
    def calc_rsi(x, window=4):
        delta = x.diff()
        u = delta.clip(lower=0)
        d = -delta.clip(upper=0)
        ewm_u = u.ewm(alpha=1/window, adjust=False).mean()
        ewm_d = d.ewm(alpha=1/window, adjust=False).mean()
        rs = ewm_u / ewm_d
        return 100 - (100 / (1 + rs))

    df['RSI_4'] = g['Close'].transform(lambda x: calc_rsi(x, window=RSI_WINDOW))
    
    # 5. ATR 40
    h = df['High']
    l = df['Low']
    pc = g['Close'].shift(1)
    
    tr1 = h - l
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['ATR_40'] = g.apply(lambda x: tr.loc[x.index].rolling(window=ATR_WINDOW).mean(), include_groups=False).reset_index(level=0, drop=True)
    
    # Shift indicators that are used for entry decisions (metrics used at EOD for Next Day Open)
    # Actually, we filter based on TODAY's Close/Indicators to enter TOMORROW Open.
    # So we don't strictly need to shift them in the dataframe if we access them by date correctly.
    # However, to avoid lookahead bias in vector operations, usually we shift.
    # But for iteration, we look at day T data to decide trade for day T+1.
    
    return df

def run_simulation(df, spy_df):
    print(f"\n>>> Simulating Strategy: Long Trend Low Volatility")
    
    # Pre-calculate SPY SMA 200
    spy_df['SMA_200'] = spy_df['Close'].rolling(window=200).mean()
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    # Align SPY to test dates
    spy_df = spy_df[spy_df.index >= start_date]
    
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    cash = STARTING_CAPITAL
    equity_curve = []
    trades = []
    positions = [] # List of dicts: Symbol, Shares, Entry_Price, Stop_Price, High_Water_Mark, Entry_Date
    
    # Stats counters
    win_count = 0
    loss_count = 0
    
    for i, date in enumerate(tqdm(sim_dates, desc="Simulating Days")):
        # We need previous day's data for SPY filter (Close > SMA200)
        # Actually, standard is "Close of S&P 500 > 200d SMA" usually means at time of signal generation.
        # If we enter Next Day Open, signal is generated on 'date'.
        
        try:
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
            
        # Get SPY data for this date
        if date not in spy_df.index:
            spy_close = 0
            spy_sma = 0
            spy_ok = False
        else:
            spy_row = spy_df.loc[date]
            spy_close = spy_row['Close']
            spy_sma = spy_row['SMA_200']
            spy_ok = (spy_close > spy_sma)
            
        remaining_positions = []
        
        # 1. Manage Existing Positions
        for pos in positions:
            symbol = pos['Symbol']
            if symbol not in day_data.index:
                remaining_positions.append(pos)
                continue
                
            row = day_data.loc[symbol]
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            exit_price = None
            exit_reason = None
            
            # Update High Water Mark (for Trailing Stop) - Check if High of today beats it? 
            # Or usually we use Close or High. Let's use High to be conservative on stop triggering?
            # Actually for trailing stop activation, if price goes up, we trail.
            # "Trailing stop of 20 percent"
            # Stop Price = High_Water_Mark * (1 - 0.20)
            # We update HWM if price moves up.
            
            # Check for Gap Down past Stop Loss
            # Fixed Stop Loss and Trailing Stop are both active.
            
            # Current effective stop is max(Fixed Stop, Trailing Stop)
            # But Fixed Stop is usually initial. Trailing takes over if it's higher.
            # Let's track both or just one dynamic stop price.
            # "Stop Loss: The day after execution, we place a stop-loss of 1.5X ATR... below execution price"
            # "Profit Protection: We also place a trailing stop of 20 percent"
            
            # Calculate current trailing stop level
            if high > pos['High_Water_Mark']:
                pos['High_Water_Mark'] = high
                
            trailing_stop_price = pos['High_Water_Mark'] * (1.0 - TRAILING_STOP_PCT)
            
            # Effective stop is the higher of the initial fixed stop and the trailing stop?
            # Or are they separate?
            # Usually trailing stop is a profit protect.
            # If trailing stop < Entry Price, and Fixed Stop < Entry Price...
            # We usually take the higher of the two to protect capital.
            
            current_stop = max(pos['Fixed_Stop_Price'], trailing_stop_price)
            
            # Check exit (Intraday Low hits stop)
            if low <= current_stop:
                # Exited
                exit_price = min(open_price, current_stop) # Slippage/Gap handling: if open < stop, exit at open
                exit_reason = "Stop Loss / Trailing Stop"
            
            if exit_price is not None:
                # Execute Exit
                gross_pnl = (exit_price - pos['Entry_Price']) * pos['Shares']
                comm_exit = max(MIN_COMMISSION, pos['Shares'] * COMMISSION_PER_SHARE)
                net_pnl = gross_pnl - comm_exit - pos['Commission_Entry']
                
                cash += (exit_price * pos['Shares']) - comm_exit
                
                trades.append({
                    'Symbol': symbol,
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
        
        # 2. Entry Logic (Enter on NEXT Open, but we calculate based on TODAY data)
        # To simulate "Next Day Market on Open", we normally queue orders.
        # However, this loop iterates 'date'. 'day_data' is Today.
        # If we signal today, we buy Tomorrow Open.
        # But 'day_data' includes Open/High/Low/Close of Today.
        # If we buy "Next Day", we need the price of Date+1.
        # Simple backtest approach: Use 'Close' of today as proxy? No, bad.
        # Better: Identify signals on Day T, execute on Day T+1 Open.
        # In this loop structure, we can just process "Pending Orders" from previous iteration.
        
        # But wait, `b_short_spike.py` used:
        # candidates = day_data[day_data['Pending_Short_Signal']]
        # where Pending_Short_Signal was shift(1).
        # Meaning: If Signal on T-1, then Pending on T.
        # Then it used `row['Open']` of T (Today) to enter. This is correct for "Next Day Open".
        
        # So we need to determine Signals for Today, and execute them Tomorrow.
        # OR, we determine Signals from Yesterday (which we can compute or shift), and execute Today.
        
        # Let's filter candidates for TODAY using logic based on TODAY's data?
        # No, if we filter on Today's data, we enter Tomorrow.
        # So we should execute orders based on YESTERDAY's signals.
        
        # Let's look for candidates from the DataFrame where we pre-calculated the Signal,
        # or calculate Signal on the fly.
        
        # Filter Conditions (On Data T-1 for Entry T)
        # But here 'day_data' is Day T.
        # We need to know if Day T-1 met the criteria.
        
        # Let's construct a "Signal" column in prepare_data?
        # Or just check conditions on `day_data` and execute tomorrow?
        # It's easier to execute "Pending Orders" from `positions` list if we had an "Orders" queue.
        # Alternatively, use the `shift` approach.
        
        # Let's compute signals in `prepare_data` or a separate pass to ensure vector speed,
        # then shift them to get "Entry_Day_Signal".
        
        pass # Logic handled in next block
        
        # Calculate Equity for Today
        curr_eq = cash
        pos_value = 0
        for p in positions:
            # Mark to Market using Today's Close
            s_sym = p['Symbol']
            if s_sym in day_data.index:
                curr_p = day_data.loc[s_sym]['Close']
            else:
                curr_p = p['Entry_Price'] # Fallback
            
            val = p['Shares'] * curr_p
            pos_value += val
            curr_eq += val
            
        equity_curve.append({
            'Date': date,
            'Equity': curr_eq,
            'Cash': cash,
            'Positions': len(positions)
        })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

# Modified run_simulation to include signal generation
def run_simulation_full(df, spy_df):
    print("Pre-calculating signals...")
    # S&P 500 Filter (Broadcast SPY SMA condition to all stocks?)
    # Easier to join or reindex SPY data.
    
    spy_df = spy_df.copy()
    spy_df['SPY_Above_SMA'] = spy_df['Close'] > spy_df['Close'].rolling(200).mean()
    spy_signal = spy_df['SPY_Above_SMA'].reindex(df.index.get_level_values('Date')).fillna(False)
    
    # We need to align this. The df index is (Date, Symbol) or (Symbol, Date)?
    # 'df' is multi-index. Let's check `prepare_data` output.
    # It sorts index. `download_and_cache_data` produces (Date, Symbol) or (Symbol, Date) depending on stack.
    # Code says: stacked.index.names = ['Date', 'Symbol']
    
    # So we can map SPY signal by Date.
    # Let's assume df is sorted by Date, Symbol
    
    # Filters
    # 1. Avg Dollar Vol > 100m
    # 2. HV between 10% and 40%
    # 3. Stock Close > SMA 200
    
    # Reset index to make boolean indexing easier or use map
    df = df.reset_index()
    df = df.set_index('Date')
    
    # Map SPY signal
    spy_series = spy_df['SPY_Above_SMA']
    df['SPY_Filter'] = df.index.map(spy_series).fillna(False)
    
    # Apply Strategy Filters
    mask = (
        (df['Avg_Dollar_Volume'] > MIN_DOLLAR_VOLUME) &
        (df['HV'] >= HV_MIN) &
        (df['HV'] <= HV_MAX) &
        (df['Close'] > df['SMA_200']) &
        (df['SPY_Filter'])
    )
    
    df['Signal'] = mask
    
    # We want to enter Next Day Open if Signal is True Today.
    # So Entry_Signal for Day T is Signal of Day T-1.
    
    # Group by Symbol to shift signal
    df = df.reset_index().set_index(['Symbol', 'Date']).sort_index()
    df['Entry_Signal'] = df.groupby(level='Symbol')['Signal'].shift(1).fillna(False)
    
    # Also need Ranking Metric (RSI 4) from T-1
    df['Rank_RSI'] = df.groupby(level='Symbol')['RSI_4'].shift(1)
    
    # Need ATR from T-1 for Stop Loss calculation at entry
    df['Entry_ATR'] = df.groupby(level='Symbol')['ATR_40'].shift(1)
    
    # Simulation Loop
    print(f"\n>>> Simulating Strategy: Long Trend Low Volatility")
    
    # Re-index for iteration by Date
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
            
        if isinstance(day_data, pd.Series): # Single symbol case
            day_data = day_data.to_frame().T
        
        # Ensure Symbol is a column if it's not (it is in reset_index)
        # If day_data has duplicate index (which it does, all same date), it's a DF.
        
        # 1. Check Stops/Exits for Existing Positions
        # We need current day's OHLC
        # day_data contains current day data for all symbols
        
        # Create a lookup for speed
        current_prices = day_data.set_index('Symbol')[['Open', 'High', 'Low', 'Close']].to_dict('index')
        
        remaining_positions = []
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
            
            # Trailing Stop Price
            trailing_stop = pos['High_Water_Mark'] * (1.0 - TRAILING_STOP_PCT)
            
            # Effective Stop
            stop_price = max(pos['Fixed_Stop_Price'], trailing_stop)
            
            exit_price = None
            exit_reason = None
            
            # Check Low against Stop
            if low_p <= stop_price:
                # Exited
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
        # Candidates: Entry_Signal is True
        candidates = day_data[day_data['Entry_Signal']].copy()
        
        # Filter out stocks already held
        held_symbols = {p['Symbol'] for p in positions}
        candidates = candidates[~candidates['Symbol'].isin(held_symbols)]
        
        if not candidates.empty:
            # Rank by Lowest RSI (4 day)
            # We want Lowest RSI.
            candidates = candidates.sort_values('Rank_RSI', ascending=True)
            
            for _, row in candidates.iterrows():
                sym = row['Symbol']
                entry_price = row['Open']
                atr = row['Entry_ATR']
                
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Position Sizing
                # Risk = 2% of Equity
                # Stop Distance = 1.5 * ATR
                # Shares = Risk_Amount / Stop_Distance
                
                current_equity = cash + sum([p['Shares'] * current_prices.get(p['Symbol'], {'Close': p['Entry_Price']})['Close'] for p in positions])
                risk_amount = current_equity * RISK_PER_TRADE
                stop_distance = STOP_LOSS_ATR_MULT * atr
                
                shares = int(risk_amount / stop_distance)
                
                # Cap Size: 10% of Equity
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
                    
                    fixed_stop = entry_price - (STOP_LOSS_ATR_MULT * atr)
                    
                    positions.append({
                        'Symbol': sym,
                        'Entry_Date': date,
                        'Entry_Price': entry_price,
                        'Shares': shares,
                        'Fixed_Stop_Price': fixed_stop,
                        'High_Water_Mark': entry_price, # Init at Entry
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
            'Drawdown': 0.0 # Calc later
        })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def main():
    symbols = get_all_symbols()
    if not symbols:
        return

    df = download_and_cache_data(symbols)
    
    # Get SPY
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS + 1) # Extra year for SMA
    spy = download_spy(start_date)
    
    df = prepare_data(df)
    
    eq_df, trades_df = run_simulation_full(df, spy)
    
    if eq_df.empty:
        print("No results generated.")
        return

    # Post-process Stats
    final_equity = eq_df.iloc[-1]['Equity']
    total_ret = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL
    
    eq_df['Daily_Log_Ret'] = np.log(eq_df['Equity'] / eq_df['Equity'].shift(1))
    eq_df['Daily_Ret'] = eq_df['Equity'].pct_change()
    
    # Sharpe Ratio (using log returns or simple? usually simple, but user asked for log returns for profit factor. Standard Sharpe uses simple excess returns usually, but log is fine too for large moves. I'll use simple for Sharpe as standard, unless implied otherwise. "sharpre ratio, profit factor (computed with daily log returns...")
    # I will use Simple Returns for Sharpe to be standard, and Log Returns for Profit Factor as requested.
    
    sharpe = (eq_df['Daily_Ret'].mean() / eq_df['Daily_Ret'].std()) * np.sqrt(252) if eq_df['Daily_Ret'].std() > 0 else 0
    
    # Profit Factor (Daily Log Returns)
    pos_log_rets = eq_df[eq_df['Daily_Log_Ret'] > 0]['Daily_Log_Ret'].sum()
    neg_log_rets = abs(eq_df[eq_df['Daily_Log_Ret'] < 0]['Daily_Log_Ret'].sum())
    profit_factor = pos_log_rets / neg_log_rets if neg_log_rets > 0 else float('inf')
    
    # Max Drawdown
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']
    max_dd = eq_df['Drawdown'].min()
    
    # Trade Stats
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
    print("RESULTS: Long Trend Low Volatility")
    print("="*60)
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Profit Factor:      {profit_factor:.2f} (Daily Log Returns)")
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Avg Win:            {avg_win_pct:.2f}%")
    print(f"Avg Loss:           {avg_loss_pct:.2f}%")
    print(f"Total Return:       {total_ret*100:.2f}%")
    print(f"Final Equity:       ${final_equity:,.2f}")
    print(f"Total Trades:       {len(trades_df)}")
    print("="*60)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(eq_df['Date'], eq_df['Equity'], label="Equity", color='blue')
    ax1.set_title("Total Equity")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.fill_between(eq_df['Date'], eq_df['Drawdown'] * 100, 0, color='red', alpha=0.3, label="Drawdown %")
    ax2.set_title("Max Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(__file__).parent / "LTlowvol_results.png"
    plt.savefig(plot_path)
    print(f"Chart saved to {plot_path}")

if __name__ == "__main__":
    main()
