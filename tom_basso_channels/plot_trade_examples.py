"""
Visualize random trades from the Tom Basso Long-Only Backtest.
Generates 30 plots of random trades showing indicators, entry/exit points, and trade stats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import random
import sys

# Add parent directory to path to import strategy if needed
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tom_basso_channels.strategy import load_stock_data
except ImportError:
    # Fallback if running from root
    from strategy import load_stock_data

# Configuration
TRADES_FILE = Path(__file__).parent / "backtest_long_only_trades.csv"
OUTPUT_DIR = Path(__file__).parent / "trade_examples"
DATA_FILE = Path(__file__).parent.parent / "mean_reversion" / "us_stock_history_10y.csv"
NUM_EXAMPLES = 30
LOOKBACK = 200
PADDING_DAYS = 50  # Days to show before entry and after exit

def calculate_bands(df):
    """Calculate the 3 channel bands for the dataframe."""
    # Donchian (200)
    # Note: Strategy uses shifted bands for signal generation (comparing Close[T] vs Band[T-1])
    # For visualization, we usually plot the band values as they exist at T-1 valid for T?
    # Or plot the bands at T corresponding to the High/Low of T-lookback..T?
    # The backtest logic: Upper[T] = Max(High[T-lookback : T-1]).
    # So at day T, the "hurdle" is the max of previous 200 days.
    # We will plot this "hurdle" value at day T.
    
    df['Donchian_Upper'] = df['High'].rolling(LOOKBACK).max().shift(1)
    df['Donchian_Lower'] = df['Low'].rolling(LOOKBACK).min().shift(1)
    
    # Keltner (200, 2.0)
    ema = df['Close'].ewm(span=LOOKBACK, adjust=False).mean()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = tr.rolling(LOOKBACK).mean()
    
    df['Keltner_Upper'] = (ema + 2.0 * atr).shift(1)
    df['Keltner_Lower'] = (ema - 2.0 * atr).shift(1)
    
    # Bollinger (200, 2.0)
    sma = df['Close'].rolling(LOOKBACK).mean()
    std = df['Close'].rolling(LOOKBACK).std()
    
    df['Bollinger_Upper'] = (sma + 2.0 * std).shift(1)
    df['Bollinger_Lower'] = (sma - 2.0 * std).shift(1)
    
    # SMA 150
    df['SMA_150'] = df['Close'].rolling(150).mean()
    
    return df

def identify_trigger(row, is_entry=True):
    """Identify which indicator triggered the signal."""
    triggers = []
    price = row['Close']
    
    if is_entry:
        # Entry: Close > Upper
        if price > row['Donchian_Upper']: triggers.append('Donchian')
        if price > row['Keltner_Upper']: triggers.append('Keltner')
        if price > row['Bollinger_Upper']: triggers.append('Bollinger')
    else:
        # Exit: Close < Lower
        if price < row['Donchian_Lower']: triggers.append('Donchian')
        if price < row['Keltner_Lower']: triggers.append('Keltner')
        if price < row['Bollinger_Lower']: triggers.append('Bollinger')
        
    return ", ".join(triggers) if triggers else "Unknown"

def plot_trade(trade, stock_data, save_path):
    """Create and save a plot for a single trade."""
    symbol = trade['Symbol']
    entry_date = pd.to_datetime(trade['Entry Date'])
    exit_date = pd.to_datetime(trade['Exit Date'])
    
    # Slice data with padding
    start_date = entry_date - pd.Timedelta(days=PADDING_DAYS)
    end_date = exit_date + pd.Timedelta(days=PADDING_DAYS)
    
    mask = (stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)
    df_slice = stock_data.loc[mask].copy()
    
    if df_slice.empty:
        print(f"No data for {symbol} in range {start_date} to {end_date}")
        return

    # Set index for plotting
    df_slice = df_slice.set_index('Date')
    
    # Create Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Price
    ax.plot(df_slice.index, df_slice['Close'], label='Close Price', color='white', linewidth=1.5)
    
    # Plot Bands (Upper)
    ax.plot(df_slice.index, df_slice['Donchian_Upper'], label='Donchian Upper', color='#ff00ff', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(df_slice.index, df_slice['Keltner_Upper'], label='Keltner Upper', color='#00ffff', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(df_slice.index, df_slice['Bollinger_Upper'], label='Bollinger Upper', color='#ffff00', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot Bands (Lower)
    ax.plot(df_slice.index, df_slice['Donchian_Lower'], color='#ff00ff', linestyle=':', alpha=0.3, linewidth=1)
    ax.plot(df_slice.index, df_slice['Keltner_Lower'], color='#00ffff', linestyle=':', alpha=0.3, linewidth=1)
    ax.plot(df_slice.index, df_slice['Bollinger_Lower'], color='#ffff00', linestyle=':', alpha=0.3, linewidth=1)
    
    # Plot SMA
    ax.plot(df_slice.index, df_slice['SMA_150'], label='SMA 150', color='orange', alpha=0.6, linewidth=1)
    
    # Highlight Trade Duration
    ax.axvspan(entry_date, exit_date, color='green', alpha=0.1, label='Trade Duration')
    
    # Mark Entry
    entry_price = trade['Entry Price']
    ax.scatter(entry_date, entry_price, color='lime', s=100, marker='^', label='Entry', zorder=5)
    
    # Mark Exit
    exit_price = trade['Exit Price']
    ax.scatter(exit_date, exit_price, color='red', s=100, marker='v', label='Exit', zorder=5)
    
    # Determine Triggers
    # We need the row data for entry date (or day before?)
    # The signal is generated on the day BEFORE the entry date (since we trade at open of entry date based on yesterday's close)
    # Wait, in the backtest logic: 
    # "action_buy" is shifted. `action_buy[T]` comes from `raw_buy[T-1]`.
    # `raw_buy[T-1]` compares `Close[T-1]` with `Bands[T-1]`.
    # So the "Trigger" happened on `Entry Date - 1 trading day`.
    # But `df_slice` has the computed bands.
    
    # Find the row for the signal date (approx 1 day before entry)
    # Since we don't have exact trading calendar here easily, we look at the row before entry_date
    try:
        entry_idx = df_slice.index.get_loc(entry_date)
        if isinstance(entry_idx, slice): entry_idx = entry_idx.start # Handle duplicate indices if any (shouldn't be)
        signal_idx = max(0, entry_idx - 1)
        signal_row = df_slice.iloc[signal_idx]
        entry_trigger = identify_trigger(signal_row, is_entry=True)
        
        # Similar for Exit
        # We exited at Open of Exit Date because "Yesterday" (Exit Date - 1) triggered sell.
        exit_idx = df_slice.index.get_loc(exit_date)
        if isinstance(exit_idx, slice): exit_idx = exit_idx.start
        exit_signal_idx = max(0, exit_idx - 1)
        exit_signal_row = df_slice.iloc[exit_signal_idx]
        exit_trigger = identify_trigger(exit_signal_row, is_entry=False)
        
    except Exception as e:
        print(f"Error identifying triggers for {symbol}: {e}")
        entry_trigger = "Unknown"
        exit_trigger = "Unknown"
        
    # Annotations
    ax.annotate(f"Entry: {entry_trigger}\n${entry_price:.2f}", 
                xy=(entry_date, entry_price), 
                xytext=(entry_date, entry_price * 0.95),
                arrowprops=dict(facecolor='lime', shrink=0.05, alpha=0.7),
                color='white', ha='center')
                
    ax.annotate(f"Exit: {exit_trigger}\n${exit_price:.2f}", 
                xy=(exit_date, exit_price), 
                xytext=(exit_date, exit_price * 1.05),
                arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),
                color='white', ha='center')

    # Info Box
    info_text = (
        f"Symbol: {symbol}\n"
        f"Return: {trade['Return %']:.2f}%\n"
        f"PnL: ${trade['PnL']:.2f}\n"
        f"Held: {int(trade['Days Held'])} days\n"
        f"Entry Trigger: {entry_trigger}\n"
        f"Exit Trigger: {exit_trigger}"
    )
    
    # Place text box in upper left
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='white')

    # Styling
    ax.set_title(f"Trade Example: {symbol} ({trade['Return %']:.2f}%)", fontsize=14, color='white')
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        
    ax.grid(True, alpha=0.1)
    ax.legend(loc='upper right', facecolor='black', edgecolor='gray', labelcolor='white')
    
    # Format Date Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    if not TRADES_FILE.exists():
        print(f"Trades file not found: {TRADES_FILE}")
        return

    print("Loading trades...")
    trades_df = pd.read_csv(TRADES_FILE)
    
    if trades_df.empty:
        print("No trades found.")
        return
        
    # Select Random Trades
    num_samples = min(NUM_EXAMPLES, len(trades_df))
    sample_trades = trades_df.sample(n=num_samples, random_state=42).to_dict('records')
    
    print(f"Selected {num_samples} trades for visualization.")
    
    # Load Stock Data
    # We load all data once, but calculating indicators for ALL symbols might be heavy if not needed.
    # However, to be safe and consistent with previous scripts, we can just load it.
    # Or better: Group by symbol and only process symbols in our sample.
    
    print("Loading stock data...")
    # Using the strategy's load function which caches or loads CSV
    full_df = load_stock_data(DATA_FILE)
    
    # Ensure types
    cols = ['Open', 'High', 'Low', 'Close']
    full_df[cols] = full_df[cols].apply(pd.to_numeric, errors='coerce')
    
    # Filter for symbols in our sample
    sample_symbols = set(t['Symbol'] for t in sample_trades)
    print(f"Processing data for {len(sample_symbols)} symbols...")
    
    df_filtered = full_df[full_df['Symbol'].isin(sample_symbols)].copy()
    
    # Group and Calculate Indicators
    # We need to calculate indicators on the full history of these symbols to avoid warmup issues in the slice
    grouped = df_filtered.groupby('Symbol')
    processed_dfs = []
    
    print("Calculating indicators...")
    for sym, group in grouped:
        group = group.sort_values('Date')
        group = calculate_bands(group)
        processed_dfs.append(group)
        
    if not processed_dfs:
        print("No data processed.")
        return
        
    indicators_df = pd.concat(processed_dfs)
    
    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    for i, trade in enumerate(sample_trades):
        symbol = trade['Symbol']
        stock_data = indicators_df[indicators_df['Symbol'] == symbol]
        
        save_path = OUTPUT_DIR / f"trade_{i+1}_{symbol}.png"
        plot_trade(trade, stock_data, save_path)
        print(f"[{i+1}/{num_samples}] Saved {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
