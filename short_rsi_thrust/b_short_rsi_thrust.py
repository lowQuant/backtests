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
MAX_POSITIONS = 10
RISK_PERCENT = 0.02  # 2% risk per trade
MAX_POSITION_SIZE_PERCENT = 0.10  # 10% max position size
HISTORY_FILE = "us_stock_history_10y.csv"
BACKTEST_YEARS = 5

# Filter Parameters
MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME_20D = 25_000_000  # $25M
MIN_ATR_PERCENT = 0.03  # 3% ATR as percentage of close

# Setup Parameters
RSI_PERIOD = 3
RSI_THRESHOLD = 90
CONSECUTIVE_UP_DAYS = 2

# Ranking
ADX_PERIOD = 7

# Entry
ENTRY_LIMIT_PERCENT = 0.04  # Sell short 4% above previous close

# Stop-Loss
ATR_PERIOD = 10
STOP_ATR_MULTIPLIER = 3.0  # 3x ATR above entry

# Profit Taking
PROFIT_TARGET_PERCENT = 0.04  # 4% profit target
MAX_HOLDING_DAYS = 2  # Exit after 2 days if no profit target hit


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
        
        # Check if local file exists in mean_reversion folder
        script_dir = Path(__file__).resolve().parent.parent / "mean_reversion"
        csv_path = script_dir / HISTORY_FILE
        if csv_path.exists():
            print(f"Local history file found at {csv_path}. Using symbols from there.")
            try:
                df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
                symbols = df.index.get_level_values(1).unique().tolist()
                print(f"Found {len(symbols)} symbols in local file.")
                return symbols
            except Exception as e2:
                print(f"Error reading local file: {e2}")

        print("Falling back to default symbol list.")
        return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL"]


def download_and_cache_data(symbols):
    script_dir = Path(__file__).resolve().parent.parent / "mean_reversion"
    csv_path = script_dir / HISTORY_FILE
    
    if csv_path.exists():
        print(f"Loading historical data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}. Re-downloading.")

    print(f"Downloading data for {len(symbols)} symbols. This may take a while...")
    
    chunk_size = 500
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    all_dfs = []

    for chunk in tqdm(chunks, desc="Downloading chunks"):
        try:
            data = yf.download(
                chunk,
                period="10y",
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


def calculate_rsi(series, period):
    """Calculate RSI for a series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(high, low, close, period):
    """Calculate ADX indicator."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx


def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def prepare_data(df):
    """
    Calculate technical indicators required for the strategy.
    """
    print("Calculating indicators...")
    
    df = df.sort_index()
    
    if "Close" not in df.columns or "Open" not in df.columns:
        df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)

    # 1. Calculate 20-day Average Dollar Volume
    print("Computing 20-day Average Dollar Volume...")
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['Avg_Dollar_Volume_20'] = df.groupby(level='Symbol')['Dollar_Volume'].transform(
        lambda x: x.rolling(window=20).mean()
    )

    # 2. Calculate 10-day ATR and ATR Percentage
    print("Computing 10-day ATR...")
    def calc_atr_group(group):
        return calculate_atr(group['High'], group['Low'], group['Close'], ATR_PERIOD)
    
    df['ATR_10'] = df.groupby(level='Symbol', group_keys=False).apply(calc_atr_group)
    df['ATR_Percent'] = df['ATR_10'] / df['Close']

    # 3. Calculate 3-day RSI
    print("Computing 3-day RSI...")
    df['RSI_3'] = df.groupby(level='Symbol')['Close'].transform(
        lambda x: calculate_rsi(x, RSI_PERIOD)
    )

    # 4. Calculate consecutive up days (close > previous close)
    print("Computing consecutive up days...")
    df['Prev_Close'] = df.groupby(level='Symbol')['Close'].shift(1)
    df['Is_Up'] = df['Close'] > df['Prev_Close']
    
    # Check if last 2 days were up
    df['Up_Day_1'] = df['Is_Up']
    df['Up_Day_2'] = df.groupby(level='Symbol')['Is_Up'].shift(1)
    df['Two_Up_Days'] = df['Up_Day_1'] & df['Up_Day_2']

    # 5. Calculate 7-day ADX for ranking
    print("Computing 7-day ADX...")
    def calc_adx_group(group):
        return calculate_adx(group['High'], group['Low'], group['Close'], ADX_PERIOD)
    
    df['ADX_7'] = df.groupby(level='Symbol', group_keys=False).apply(calc_adx_group)

    return df


def run_backtest(df):
    """
    Run the short RSI thrust backtest.
    
    Strategy:
    - Filter: Price >= $5, Avg Dollar Volume 20d > $25M, ATR% >= 3%
    - Setup: RSI(3) > 90, last 2 days close > previous close
    - Rank by: Highest ADX(7)
    - Entry: Sell short at 4% above previous close (limit order)
    - Stop: 3x ATR(10) above entry price
    - Profit: Exit at MOC if profit >= 4%, or exit after 2 days
    - Position sizing: 2% risk, 10% max size, max 10 positions
    """
    print("\n>>> Running Short RSI Thrust Backtest")
    
    dates = df.index.get_level_values('Date').unique().sort_values()
    start_date = dates.max() - pd.DateOffset(years=BACKTEST_YEARS)
    
    test_df = df[df.index.get_level_values('Date') >= start_date].copy()
    
    sim_dates = test_df.index.get_level_values('Date').unique().sort_values()
    
    # Initialize tracking
    # For short positions, we track:
    # - cash: available cash (starts at STARTING_CAPITAL)
    # - margin_used: cash tied up as margin for short positions
    # - realized_pnl: cumulative realized P&L from closed trades
    cash = STARTING_CAPITAL
    realized_pnl = 0
    equity_curve = []
    trades = []
    open_positions = {}  # symbol -> {entry_price, entry_date, stop_price, shares, days_held, margin}
    pending_orders = {}  # symbol -> {limit_price, signal_date, atr}
    
    for i, date in enumerate(tqdm(sim_dates, desc="Simulating")):
        try:
            day_data = test_df.xs(date, level='Date')
        except KeyError:
            continue
        
        # Track daily P&L for open positions
        daily_pnl = 0
        positions_to_close = []
        
        # 1. Check existing positions for stop-loss, profit target, or time exit
        for symbol, pos in open_positions.items():
            if symbol not in day_data.index:
                continue
                
            row = day_data.loc[symbol]
            high_price = row['High']
            close_price = row['Close']
            
            pos['days_held'] += 1
            
            # Check stop-loss (buy stop triggered if high >= stop price)
            if high_price >= pos['stop_price']:
                # Stopped out - assume filled at stop price
                exit_price = pos['stop_price']
                pnl = (pos['entry_price'] - exit_price) * pos['shares']
                daily_pnl += pnl
                trades.append({
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': date,
                    'Symbol': symbol,
                    'Entry_Price': pos['entry_price'],
                    'Exit_Price': exit_price,
                    'Shares': pos['shares'],
                    'PnL': pnl,
                    'Return': (pos['entry_price'] - exit_price) / pos['entry_price'],
                    'Exit_Reason': 'Stop-Loss'
                })
                positions_to_close.append(symbol)
                continue
            
            # Check profit target at close (4% profit)
            profit_pct = (pos['entry_price'] - close_price) / pos['entry_price']
            if profit_pct >= PROFIT_TARGET_PERCENT:
                # Exit at MOC next day - mark for exit
                pos['exit_at_moc'] = True
                continue
            
            # Check time-based exit (after 2 days)
            if pos['days_held'] >= MAX_HOLDING_DAYS:
                pos['exit_at_moc'] = True
                continue
        
        # Remove stopped out positions - release margin
        for symbol in positions_to_close:
            cash += open_positions[symbol]['margin']  # Release margin
            del open_positions[symbol]
        
        # 2. Execute MOC exits from previous day's signals
        positions_to_close = []
        for symbol, pos in open_positions.items():
            if pos.get('exit_at_moc'):
                if symbol not in day_data.index:
                    continue
                row = day_data.loc[symbol]
                exit_price = row['Close']
                pnl = (pos['entry_price'] - exit_price) * pos['shares']
                daily_pnl += pnl
                trades.append({
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': date,
                    'Symbol': symbol,
                    'Entry_Price': pos['entry_price'],
                    'Exit_Price': exit_price,
                    'Shares': pos['shares'],
                    'PnL': pnl,
                    'Return': (pos['entry_price'] - exit_price) / pos['entry_price'],
                    'Exit_Reason': 'Profit Target' if (pos['entry_price'] - exit_price) / pos['entry_price'] >= PROFIT_TARGET_PERCENT else 'Time Exit'
                })
                positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            cash += open_positions[symbol]['margin']  # Release margin
            del open_positions[symbol]
        
        # Update realized P&L
        realized_pnl += daily_pnl
        
        # 3. Try to fill pending short orders
        orders_to_remove = []
        for symbol, order in pending_orders.items():
            if symbol not in day_data.index:
                orders_to_remove.append(symbol)
                continue
            
            if symbol in open_positions:
                orders_to_remove.append(symbol)
                continue
            
            row = day_data.loc[symbol]
            high_price = row['High']
            open_price = row['Open']
            limit_price = order['limit_price']
            
            entry_price = None
            
            # Check if limit order fills (price goes up to our limit)
            if open_price >= limit_price:
                # Gap up - fill at open
                entry_price = open_price
            elif high_price >= limit_price:
                # Intraday fill at limit
                entry_price = limit_price
            
            if entry_price:
                # Calculate position size based on risk
                stop_price = entry_price + (STOP_ATR_MULTIPLIER * order['atr'])
                risk_per_share = stop_price - entry_price
                
                current_equity = cash + sum(
                    pos['entry_price'] * pos['shares'] for pos in open_positions.values()
                )
                
                # Risk-based sizing
                risk_amount = current_equity * RISK_PERCENT
                shares_by_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                
                # Max position size constraint
                max_position_value = current_equity * MAX_POSITION_SIZE_PERCENT
                shares_by_size = int(max_position_value / entry_price) if entry_price > 0 else 0
                
                # Take minimum
                shares = min(shares_by_risk, shares_by_size)
                
                # Check if we have capacity for new position
                if len(open_positions) >= MAX_POSITIONS:
                    orders_to_remove.append(symbol)
                    continue
                
                # Margin requirement for short position (use position value as margin)
                margin_required = entry_price * shares
                
                if shares > 0 and cash >= margin_required:
                    # Open short position - deduct margin from cash
                    cash -= margin_required
                    
                    open_positions[symbol] = {
                        'entry_price': entry_price,
                        'entry_date': date,
                        'stop_price': stop_price,
                        'shares': shares,
                        'days_held': 0,
                        'exit_at_moc': False,
                        'margin': margin_required
                    }
                    
            orders_to_remove.append(symbol)
        
        for symbol in orders_to_remove:
            if symbol in pending_orders:
                del pending_orders[symbol]
        
        # 4. Generate new signals for tomorrow
        # Apply filters
        candidates = day_data[
            (day_data['Close'] >= MIN_PRICE) &
            (day_data['Avg_Dollar_Volume_20'] >= MIN_AVG_DOLLAR_VOLUME_20D) &
            (day_data['ATR_Percent'] >= MIN_ATR_PERCENT) &
            (day_data['RSI_3'] > RSI_THRESHOLD) &
            (day_data['Two_Up_Days'] == True)
        ].copy()
        
        if not candidates.empty:
            # Rank by highest ADX
            candidates = candidates.sort_values('ADX_7', ascending=False)
            
            # Take top candidates (up to max positions available)
            available_slots = MAX_POSITIONS - len(open_positions)
            top_candidates = candidates.head(available_slots)
            
            for symbol in top_candidates.index:
                if symbol in open_positions or symbol in pending_orders:
                    continue
                
                row = top_candidates.loc[symbol]
                limit_price = row['Close'] * (1 + ENTRY_LIMIT_PERCENT)
                
                pending_orders[symbol] = {
                    'limit_price': limit_price,
                    'signal_date': date,
                    'atr': row['ATR_10']
                }
        
        # Calculate current equity
        # Equity = Cash + Margin in positions + Unrealized P&L
        unrealized_pnl = 0
        total_margin = 0
        for symbol, pos in open_positions.items():
            total_margin += pos['margin']
            if symbol in day_data.index:
                current_price = day_data.loc[symbol]['Close']
                # Short position P&L: (entry - current) * shares
                unrealized_pnl += (pos['entry_price'] - current_price) * pos['shares']
        
        current_equity = cash + total_margin + unrealized_pnl + realized_pnl
        
        equity_curve.append({
            'Date': date,
            'Equity': current_equity,
            'Cash': cash,
            'Positions': len(open_positions),
            'Pending_Orders': len(pending_orders)
        })
    
    return equity_curve, trades


def calculate_stats(equity_curve, trades):
    """Calculate backtest statistics."""
    if not equity_curve:
        return None
    
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
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
        
        # Average holding period
        if 'Entry_Date' in trades_df.columns and 'Exit_Date' in trades_df.columns:
            trades_df['Holding_Days'] = (pd.to_datetime(trades_df['Exit_Date']) - 
                                          pd.to_datetime(trades_df['Entry_Date'])).dt.days
            avg_holding = trades_df['Holding_Days'].mean()
        else:
            avg_holding = 0
            
        # Exit reason breakdown
        exit_reasons = trades_df['Exit_Reason'].value_counts().to_dict() if 'Exit_Reason' in trades_df.columns else {}
    else:
        win_rate = 0
        avg_trade_ret = 0
        trade_count = 0
        pf = 0
        avg_holding = 0
        exit_reasons = {}
    
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_dd = equity_df['Drawdown'].min() * 100
    
    stats = {
        "Final Equity": final_equity,
        "Total Return %": total_return,
        "Sharpe": sharpe,
        "Trades": trade_count,
        "Win Rate %": win_rate,
        "Avg Trade %": avg_trade_ret,
        "Max DD %": max_dd,
        "Profit Factor": pf,
        "Avg Holding Days": avg_holding,
        "Exit Reasons": exit_reasons
    }
    
    return stats, equity_df, trades_df


def main():
    symbols = get_all_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        return

    df = download_and_cache_data(symbols)
    df = prepare_data(df)
    
    equity_curve, trades = run_backtest(df)
    
    stats, equity_df, trades_df = calculate_stats(equity_curve, trades)
    
    script_dir = Path(__file__).resolve().parent
    
    # Print Results
    print("\n" + "="*60)
    print("SHORT RSI THRUST BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital:    ${STARTING_CAPITAL:,.2f}")
    print(f"Final Equity:        ${stats['Final Equity']:,.2f}")
    print(f"Total Return:        {stats['Total Return %']:.2f}%")
    print(f"Sharpe Ratio:        {stats['Sharpe']:.2f}")
    print(f"Max Drawdown:        {stats['Max DD %']:.2f}%")
    print(f"Total Trades:        {stats['Trades']}")
    print(f"Win Rate:            {stats['Win Rate %']:.2f}%")
    print(f"Avg Trade Return:    {stats['Avg Trade %']:.2f}%")
    print(f"Profit Factor:       {stats['Profit Factor']:.2f}")
    print(f"Avg Holding Days:    {stats['Avg Holding Days']:.1f}")
    print("-"*60)
    print("Exit Reasons:")
    for reason, count in stats['Exit Reasons'].items():
        print(f"  {reason}: {count}")
    print("="*60)
    
    # Save results
    equity_df.to_csv(script_dir / "equity_curve.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(script_dir / "trades.csv", index=False)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df['Date'], equity_df['Equity'], label='Equity')
    plt.title("Short RSI Thrust - Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(script_dir / "equity_curve.png")
    print(f"\nEquity curve saved to {script_dir / 'equity_curve.png'}")
    
    # Plot drawdown
    plt.figure(figsize=(12, 4))
    plt.fill_between(equity_df['Date'], equity_df['Drawdown'] * 100, 0, alpha=0.3, color='red')
    plt.plot(equity_df['Date'], equity_df['Drawdown'] * 100, color='red', linewidth=0.5)
    plt.title("Short RSI Thrust - Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(script_dir / "drawdown.png")
    print(f"Drawdown chart saved to {script_dir / 'drawdown.png'}")


if __name__ == "__main__":
    main()
