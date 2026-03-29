"""
Tom Basso Channel Strategy - Long Only with Benchmark & Detailed Reporting

Backtest settings:
- Capital: $100,000
- Position Size: 5%
- Commission: $0.005/share (min $1)
- Filter: Top 100 Dollar Volume
- Trend Filter: Close > 150d SMA
- Signals: Donchian (200), Keltner (200, 2.0), Bollinger (200, 2.0)
- Benchmark: SPY

Outputs:
- Equity Curve vs SPY (plus Drawdown & Open Positions subplots)
- Trades CSV with indicator values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from strategy import donchian_signal, keltner_signal, bollinger_signal, load_stock_data
except ModuleNotFoundError:
    from tom_basso_channels.strategy import donchian_signal, keltner_signal, bollinger_signal, load_stock_data

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 100_000
POSITION_SIZE_PCT = 0.05
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0
LOOKBACK = 200
SMA_FILTER_PERIOD = 150
TOP_N_STOCKS = 100

def calculate_commission(shares: float, price: float) -> float:
    commission = abs(shares) * COMMISSION_PER_SHARE
    return max(commission, MIN_COMMISSION)

def filter_top_traded_stocks(df: pd.DataFrame, top_n: int = 100) -> set:
    df['dollar_volume'] = df['Volume'] * df['Close']
    avg_dollar_vol = df.groupby('Symbol')['dollar_volume'].mean().sort_values(ascending=False)
    return set(avg_dollar_vol.head(top_n).index)

def calculate_indicators_for_logging(grp):
    """Calculate indicator values for logging purposes."""
    # Donchian
    grp['Donchian_Upper'] = grp['High'].rolling(LOOKBACK).max().shift(1)
    grp['Donchian_Lower'] = grp['Low'].rolling(LOOKBACK).min().shift(1)
    
    # Keltner
    ema = grp['Close'].ewm(span=LOOKBACK, adjust=False).mean()
    high_low = grp['High'] - grp['Low']
    high_close = np.abs(grp['High'] - grp['Close'].shift(1))
    low_close = np.abs(grp['Low'] - grp['Close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = tr.rolling(LOOKBACK).mean()
    grp['Keltner_Upper'] = (ema + 2.0 * atr).shift(1)
    grp['Keltner_Lower'] = (ema - 2.0 * atr).shift(1)
    
    # Bollinger
    sma = grp['Close'].rolling(LOOKBACK).mean()
    std = grp['Close'].rolling(LOOKBACK).std()
    grp['Bollinger_Upper'] = (sma + 2.0 * std).shift(1)
    grp['Bollinger_Lower'] = (sma - 2.0 * std).shift(1)
    
    # SMA Filter
    grp['SMA_150'] = grp['Close'].rolling(SMA_FILTER_PERIOD).mean()
    
    return grp

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing data and calculating indicators...")
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Apply indicator calculation per group
    # We do this to have the values ready for the trade log
    # This might be memory intensive, but necessary for the requirement
    tqdm.pandas(desc="Calculating Indicators")
    df = df.groupby('Symbol', group_keys=False).progress_apply(calculate_indicators_for_logging)
    
    # Calculate Signals
    # Buy if Close > ANY Upper Band AND Close > SMA_150
    # Sell if Close < ANY Lower Band
    
    # Check Buy Conditions
    buy_cond = (
        ((df['Close'] > df['Donchian_Upper']) | 
         (df['Close'] > df['Keltner_Upper']) | 
         (df['Close'] > df['Bollinger_Upper'])) &
        (df['Close'] > df['SMA_150'])
    )
    
    # Check Sell Conditions
    sell_cond = (
        (df['Close'] < df['Donchian_Lower']) | 
        (df['Close'] < df['Keltner_Lower']) | 
        (df['Close'] < df['Bollinger_Lower'])
    )
    
    # Generate Signal Series: 1 (Long), 0 (Flat/Exit)
    # Note: This is a simplifiction. The original strategy logic holds until sell signal.
    # We need to construct the stateful signal.
    
    df['raw_buy'] = buy_cond
    df['raw_sell'] = sell_cond
    
    # We will handle the stateful logic in the event loop for precision
    
    # Dollar Volume for ranking
    df['dollar_volume'] = df['Volume'] * df['Close']
    
    return df

def load_spy_benchmark():
    path = Path(__file__).parent.parent / "data" / "SPY_20150102_20251230.parquet"
    if not path.exists():
        print(f"Warning: SPY benchmark file not found at {path}")
        return None
    
    df = pd.read_parquet(path)
    # Parquet index might be named 'timestamp' or 'Date' or just be the index
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('Date')
    elif not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)
    
    df = df.sort_index()
    # Normalize column names
    df = df.rename(columns=str.lower)
    return df['close']

def run_backtest(df: pd.DataFrame):
    print("\nRunning Long-Only Backtest...")
    
    dates = sorted(df['Date'].unique())
    cash = STARTING_CAPITAL
    positions = {} # {symbol: {shares, entry_price, entry_date}}
    
    equity_curve = []
    trades = []
    
    # Pre-index data for speed
    df_indexed = df.set_index(['Date', 'Symbol']).sort_index()
    
    for date in tqdm(dates, desc="Simulation"):
        try:
            day_data = df_indexed.loc[date]
        except KeyError:
            continue
            
        # 1. Update Portfolio Value & Check Exits
        current_equity = cash
        current_value = 0
        symbols_to_exit = []
        
        # We need to iterate over positions to update value and check exits
        # Using a list to avoid runtime modification issues
        active_symbols = list(positions.keys())
        
        for sym in active_symbols:
            if sym not in day_data.index:
                # Stock delisted or missing data? Assume price didn't change (or use last known)
                # For simplicity in this robust backtest, we skip update if data missing
                # checking exits might be impossible without data.
                continue
                
            row = day_data.loc[sym]
            price = row['Close']
            
            # Update Equity
            pos = positions[sym]
            pos_val = pos['shares'] * price
            current_value += pos_val
            
            # Check Exit Signal
            # Sell if Close < Lower Band of ANY indicator
            if row['raw_sell']:
                symbols_to_exit.append(sym)
        
        current_equity = cash + current_value
        
        # Execute Exits
        for sym in symbols_to_exit:
            row = day_data.loc[sym]
            price = row['Open'] # Exit at Open of THIS day? 
            # Strategy says: "Exit: Close position next day if ANY indicator reverses"
            # Our 'raw_sell' is based on Close. So if Close < Lower Band yesterday?
            # Wait, the prompt says "Entry: Buy next day if ANY indicator signals a buy... Exit: Close position next day if ANY indicator reverses"
            # The 'raw_sell' calculated in prepare_data uses current day's Close vs Shifted Bands.
            # Bands are shifted 1. So 'Donchian_Lower' at T is calculated from T-1 data.
            # If Close(T) < Lower(T), that is the breakdown.
            # We should exit at T+1 Open.
            
            # BUT, in this loop, 'date' is T.
            # If we detected a breakdown at T-1, we should exit at T Open.
            # To simulate this correctly:
            # We need to know if we had a sell signal YESTERDAY.
            # The 'raw_sell' column tells us if the condition is met TODAY.
            
            # Let's adjust logic:
            # We need to track if a position is "pending exit".
            # Or simpler: The signal is generated at Close(T). We trade at Open(T+1).
            # So, inside the loop (Day T), we look at signals from Day T-1.
            # Actually, `prepare_data` calculates `raw_sell` based on current close vs shifted bands.
            # So `raw_sell` at T means "Close(T) < Band(determined by T-1)".
            # This is the breakdown event. We should exit T+1.
            
            # HOWEVER, the `backtest_portfolio.py` used `signal` which was pre-calculated.
            # Here we have `raw_sell` computed on the row.
            
            # Let's check `backtest_portfolio.py` logic again.
            # It checks `should_exit`.
            # If `signal != 1` (for long). `signal` was constructed from `get_combined_signal`.
            # `get_combined_signal` uses `shift(1)` for bands.
            # So `signal[T]` depends on `Close[T]` vs `Band[T]`.
            # Wait, `get_combined_signal` logic:
            # `upper = high.rolling(lookback).max().shift(1)`
            # `signal.loc[close > upper] = 1`
            # So `signal[T]` is 1 if `Close[T] > Max(High[T-lookback : T-1])`.
            # This is a breakout AT T.
            # We trade at Open T+1.
            
            # So, in the loop (Day T), we should check signals from T-1?
            # Or simpler: The loop processes Day T.
            # We have open positions.
            # We check if they triggered a sell signal YESTERDAY (T-1).
            # If yes, we sell at Open(T).
            pass
        
        # To do this cleanly, we need to access previous day's data or store the exit flag on the position.
        # Let's store "signal_date" in the dataframe? No.
        # Let's look up the signal from the dataframe.
        # But we don't have easy random access to T-1 in this loop structure without index manipulation.
        
        # ALTERNATIVE:
        # In `prepare_data`, we can shift the signals by 1.
        # `df['trade_signal'] = df['raw_buy'].shift(1)`
        # `df['exit_signal'] = df['raw_sell'].shift(1)`
        # Then at Day T, if `exit_signal` is True, it means T-1 Close broke down. We exit at T Open.
        # This simplifies the loop immensely.
        
        pass 
    
    # Re-doing the signal shift in prepare_data would be best.
    # But I can't easily modify prepare_data inside run_backtest and I don't want to rewrite prepare_data completely if I can help it.
    # Actually, I can just do the shift in `run_backtest` before the loop.
    
    # Let's Refine `run_backtest`
    
    # Group by symbol and shift signals
    df['action_buy'] = df.groupby('Symbol')['raw_buy'].shift(1).fillna(False)
    df['action_sell'] = df.groupby('Symbol')['raw_sell'].shift(1).fillna(False)
    
    df_indexed = df.set_index(['Date', 'Symbol']).sort_index()
    
    for date in tqdm(dates, desc="Simulation"):
        try:
            day_data = df_indexed.loc[date]
        except KeyError:
            continue
            
        current_value = 0
        symbols_to_exit = []
        
        # 1. Process Exits (Sell at Open)
        for sym, pos in list(positions.items()):
            if sym not in day_data.index:
                continue
            
            row = day_data.loc[sym]
            
            # If action_sell is True, it means yesterday closed below bands
            if row['action_sell']:
                exit_price = row['Open']
                shares = pos['shares']
                
                # Commission
                comm = calculate_commission(shares, exit_price)
                proceeds = (shares * exit_price) - comm
                cash += proceeds
                
                # Log Trade
                pnl = proceeds - (pos['cost_basis']) # cost_basis includes entry comm
                ret_pct = (pnl / pos['cost_basis']) * 100
                days_held = int((date - pos['entry_date']) / np.timedelta64(1, 'D'))
                
                trades.append({
                    'Symbol': sym,
                    'Entry Date': pos['entry_date'],
                    'Exit Date': date,
                    'Entry Price': pos['entry_price'],
                    'Exit Price': exit_price,
                    'Shares': shares,
                    'PnL': pnl,
                    'Return %': ret_pct,
                    'Days Held': days_held,
                    'Entry Donchian': pos['indicators']['Donchian_Upper'],
                    'Exit Donchian': row['Donchian_Lower'],
                    'Entry Keltner': pos['indicators']['Keltner_Upper'],
                    'Exit Keltner': row['Keltner_Lower'],
                    'Entry BB': pos['indicators']['Bollinger_Upper'],
                    'Exit BB': row['Bollinger_Lower'],
                    'Exit Reason': 'Signal'
                })
                
                del positions[sym]
            else:
                # Update Value
                current_value += pos['shares'] * row['Close']

        current_equity = cash + current_value
        
        # 2. Process Entries (Buy at Open)
        # Identify candidates
        # Candidates are those with action_buy == True
        # And not currently in positions
        
        candidates = day_data[day_data['action_buy'] & ~day_data.index.isin(positions.keys())]
        
        if not candidates.empty:
            # Sort by Dollar Volume
            candidates = candidates.sort_values('dollar_volume', ascending=False)
            
            target_pos_size = current_equity * POSITION_SIZE_PCT
            
            for sym, row in candidates.iterrows():
                buy_price = row['Open']
                if pd.isna(buy_price) or buy_price <= 0:
                    continue
                    
                shares = int(target_pos_size / buy_price)
                if shares <= 0:
                    continue
                
                comm = calculate_commission(shares, buy_price)
                cost = (shares * buy_price) + comm
                
                if cash >= cost:
                    cash -= cost
                    positions[sym] = {
                        'shares': shares,
                        'entry_price': buy_price,
                        'entry_date': date,
                        'cost_basis': cost,
                        'indicators': {
                            'Donchian_Upper': row['Donchian_Upper'],
                            'Keltner_Upper': row['Keltner_Upper'],
                            'Bollinger_Upper': row['Bollinger_Upper']
                        }
                    }
                    current_value += shares * row['Close'] # Mark to market at close
        
        current_equity = cash + current_value
        
        # Record Daily Stats
        equity_curve.append({
            'Date': date,
            'Equity': current_equity,
            'Cash': cash,
            'Positions': len(positions)
        })

    # Close all at end
    last_date = dates[-1]
    if positions:
        try:
            day_data = df_indexed.loc[last_date]
            for sym, pos in list(positions.items()):
                if sym in day_data.index:
                    row = day_data.loc[sym]
                    exit_price = row['Close']
                    shares = pos['shares']
                    comm = calculate_commission(shares, exit_price)
                    proceeds = (shares * exit_price) - comm
                    pnl = proceeds - pos['cost_basis']
                    ret_pct = (pnl / pos['cost_basis']) * 100
                    days_held = int((last_date - pos['entry_date']) / np.timedelta64(1, 'D'))
                    
                    trades.append({
                        'Symbol': sym,
                        'Entry Date': pos['entry_date'],
                        'Exit Date': last_date,
                        'Entry Price': pos['entry_price'],
                        'Exit Price': exit_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return %': ret_pct,
                        'Days Held': days_held,
                        'Entry Donchian': pos['indicators']['Donchian_Upper'],
                        'Exit Donchian': row['Donchian_Lower'],
                        'Entry Keltner': pos['indicators']['Keltner_Upper'],
                        'Exit Keltner': row['Keltner_Lower'],
                        'Entry BB': pos['indicators']['Bollinger_Upper'],
                        'Exit BB': row['Bollinger_Lower'],
                        'Exit Reason': 'End of Backtest'
                    })
        except KeyError:
            pass

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def main():
    print("Loading Data...")
    df = load_stock_data()
    # Ensure numerical types for calc
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=cols)
    
    # Filter Universe First to speed up indicator calc?
    # No, we need indicators for all because ranking changes.
    # But we can limit to top 500 or something if it's too slow.
    # For now, let's process all.
    
    # But wait, filtering to top 100 is dynamic in the strategy usually?
    # The previous `backtest_portfolio.py` filtered the universe upfront to top 100 *average* dollar volume.
    # "Filter to top 100 most traded stocks..."
    # If we stick to that logic (static universe), it's faster.
    
    print("Filtering Universe (Static Top 100)...")
    top_stocks = filter_top_traded_stocks(df, TOP_N_STOCKS)
    df = df[df['Symbol'].isin(top_stocks)].copy()
    
    df = prepare_data(df)
    
    equity_df, trades_df = run_backtest(df)
    
    # Benchmark
    print("Loading Benchmark...")
    spy = load_spy_benchmark()
    
    # Analysis & Plotting
    equity_df['Date'] = pd.to_datetime(equity_df['Date'])
    equity_df = equity_df.set_index('Date')
    
    # Align SPY
    if spy is not None:
        spy = spy.reindex(equity_df.index).fillna(method='ffill')
        # Normalize
        spy_norm = spy / spy.iloc[0] * STARTING_CAPITAL
    
    # Calculate Drawdown
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    
    # Plotting
    print("Generating Plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. Equity
    ax1.plot(equity_df.index, equity_df['Equity'], label='Tom Basso Long Only', color='#00ff00')
    if spy is not None:
        ax1.plot(spy_norm.index, spy_norm, label='SPY Benchmark', color='gray', alpha=0.7)
    ax1.set_title('Equity Curve vs SPY')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2.fill_between(equity_df.index, equity_df['Drawdown'] * 100, 0, color='red', alpha=0.3)
    ax2.plot(equity_df.index, equity_df['Drawdown'] * 100, color='red', linewidth=1)
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('%')
    ax2.grid(True, alpha=0.3)
    
    # 3. Positions
    ax3.bar(equity_df.index, equity_df['Positions'], width=1.0, color='#00ccff', alpha=0.7)
    ax3.set_title('Number of Open Positions')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("tom_basso_channels/backtest_long_only_result.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Save Trades
    trades_path = Path("tom_basso_channels/backtest_long_only_trades.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"Trades saved to {trades_path}")
    
    # Print Summary
    total_ret = (equity_df['Equity'].iloc[-1] / STARTING_CAPITAL) - 1
    cagr = (1 + total_ret) ** (365 / (equity_df.index[-1] - equity_df.index[0]).days) - 1
    max_dd = equity_df['Drawdown'].min()
    
    print("\nSummary Results:")
    print(f"Final Equity: ${equity_df['Equity'].iloc[-1]:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Total Trades: {len(trades_df)}")

if __name__ == "__main__":
    main()
