import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import b_short_spike_jp as strategy

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STYLE_CONFIG = {
    'background': '#0d1117', # Github Dark Dimmed-ish
    'grid': '#30363d',
    'text': '#c9d1d9',
    'bull_color': '#238636', # Green
    'bear_color': '#da3633', # Red
    'candle_width': 0.6,
    'volume_alpha': 0.5
}

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = STYLE_CONFIG['background']
    plt.rcParams['axes.facecolor'] = STYLE_CONFIG['background']
    plt.rcParams['grid.color'] = STYLE_CONFIG['grid']
    plt.rcParams['text.color'] = STYLE_CONFIG['text']
    plt.rcParams['axes.labelcolor'] = STYLE_CONFIG['text']
    plt.rcParams['xtick.color'] = STYLE_CONFIG['text']
    plt.rcParams['ytick.color'] = STYLE_CONFIG['text']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def draw_candlesticks(ax, data):
    width = STYLE_CONFIG['candle_width']
    
    # Need numerical index for width logic in Matplotlib, or strict Date handling
    # We will use the matplotlib dates
    
    # Convert dates to mdates if not already
    dates = mdates.date2num(data.index.to_pydatetime())
    
    opens = data['Open'].values
    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    
    # Iterate and draw
    for i in range(len(data)):
        d = dates[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        color = STYLE_CONFIG['bull_color'] if c >= o else STYLE_CONFIG['bear_color']
        
        # High-Low Line
        ax.plot([d, d], [l, h], color=color, linewidth=1, zorder=1)
        
        # Open-Close Rectangle
        height = abs(c - o)
        # Minimum height for doji
        if height == 0:
            height = (h-l) * 0.05 if h!=l else 0.01
            
        y = min(o, c)
        
        rect = Rectangle(
            xy=(d - width/2, y),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
            zorder=2
        )
        ax.add_patch(rect)

def plot_trade(trade, daily_data, output_dir, trade_idx=1):
    symbol = trade['Symbol']
    entry_date = trade['Entry_Date']
    exit_date = trade['Exit_Date']
    
    # Define window: Increased to 60 days to push entry to the right and avoid legend overlap
    start_plot = entry_date - pd.Timedelta(days=60) 
    end_plot = exit_date + pd.Timedelta(days=15)
    
    try:
        data = daily_data.xs(symbol, level='Symbol')
    except KeyError:
        return

    subset = data[(data.index >= start_plot) & (data.index <= end_plot)].copy()
    if subset.empty: return
    
    # Get Signal Day Data (Day before Entry)
    # Finding the row in 'data' (not subset) to be safe
    try:
        # Assuming entry is next trading day, look for day immediately preceding entry
        # We can find the location of entry_date and subtract 1
        idx_loc = data.index.get_loc(entry_date)
        if idx_loc > 0:
            signal_day = data.iloc[idx_loc - 1]
            signal_date = data.index[idx_loc - 1]
            
            # Extract Metrics
            ret_3d = signal_day.get('Ret_3D', 0)
            rsi = signal_day.get('RSI', 0)
            adx = signal_day.get('ADX', 0)
        else:
            ret_3d, rsi, adx = 0, 0, 0
            signal_date = entry_date
    except Exception:
         ret_3d, rsi, adx = 0, 0, 0
         signal_date = entry_date

    # Setup Figure
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})
    
    # 1. Main Chart (Candles)
    draw_candlesticks(ax1, subset)
    
    # Indicators
    if 'EMA_Exit' in subset.columns:
        ax1.plot(subset.index, subset['EMA_Exit'], color='#e3b341', linestyle='-', linewidth=1, alpha=0.8, label='EMA(5)')

    # Markers
    entry_price = trade['Entry_Price']
    exit_price = trade['Exit_Price']
    
    # Entry Marker
    ax1.plot(entry_date, entry_price, marker='v', markersize=12, color='#f0883e', markeredgecolor='white', zorder=10)

    # Exit Marker
    color_exit = '#238636' if trade['PnL'] > 0 else '#da3633'
    ax1.plot(exit_date, exit_price, marker='^', markersize=12, color=color_exit, markeredgecolor='white', zorder=10)

    # Info Box Legend
    info_text = (
        f"TRADE #{trade_idx} DETAILS\n"
        f"──────────────────────────\n"
        f"Symbol:     {symbol}\n"
        f"Entry Date: {entry_date.strftime('%Y-%m-%d')}\n"
        f"Entry Px:   ¥{entry_price:,.0f}\n"
        f"Signal Day: {signal_date.strftime('%Y-%m-%d')}\n"
        f"  Move 3D:  {ret_3d*100:+.1f}%\n"
        f"  RSI(4):   {rsi:.1f}\n"
        f"  ADX(14):  {adx:.1f}\n"
        f"──────────────────────────\n"
        f"Exit Date:  {exit_date.strftime('%Y-%m-%d')}\n"
        f"Exit Px:    ¥{exit_price:,.0f}\n"
        f"Return:     {trade['Return']*100:+.2f}%\n"
        f"PnL:        ¥{trade['PnL']:,.0f}\n"
        f"Reason:     {trade['Reason']}"
    )
    
    # Place text box in upper left (or best fit)
    props = dict(boxstyle='round', facecolor=STYLE_CONFIG['background'], alpha=0.9, edgecolor=STYLE_CONFIG['grid'])
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    # Title
    ax1.set_title(f"Trade {symbol} | {trade['Return']*100:+.2f}% | {trade['Reason']}", 
                  fontsize=14, fontweight='bold', pad=20, color='white')
    
    ax1.grid(True, linestyle='--', alpha=0.2)
    # Legend for lines only
    ax1.legend(loc='upper right', frameon=False)
    
    # 2. Volume Chart
    colors = [STYLE_CONFIG['bull_color'] if c >= o else STYLE_CONFIG['bear_color'] 
              for c, o in zip(subset['Close'], subset['Open'])]
    ax2.bar(subset.index, subset['Volume'], color=colors, alpha=0.6, width=0.6)
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    # Formatting X-Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(subset)//15)))
    plt.xticks(rotation=45, ha='right')
    
    # Save
    filename = output_dir / f"trade_{symbol}_{entry_date.strftime('%Y%m%d')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    print("Loading data...")
    symbols = strategy.get_all_symbols()
    if not symbols: return

    df = strategy.download_and_cache_data(symbols)
    if df.empty: return
        
    df = strategy.prepare_data(df)
    
    # Run Simulation (New Dynamic Logic)
    print("Running simulation...")
    # Passing 0.0 as size since new logic ignores it, but function signature expects it
    eq_df, trades_df, skipped, exits = strategy.run_simulation(df, 0.0)
    
    if trades_df.empty:
        print("No trades.")
        return
        
    output_dir = strategy.Path("example_trades")
    output_dir.mkdir(exist_ok=True)
    
    # Generate 100 random trades
    num_trades = min(100, len(trades_df))
    print(f"Generating {num_trades} random example trades...")
    
    # Select random sample
    random_trades = trades_df.sample(n=num_trades, random_state=42)
    
    for i, (_, trade) in enumerate(random_trades.iterrows(), 1):
        print(f"[{i}/{num_trades}] Plotting {trade['Symbol']}...")
        plot_trade(trade, df, output_dir, trade_idx=i)

    print("Done.")

if __name__ == "__main__":
    main()