"""
Tom Basso Channel Strategy - Portfolio Backtest

Proper day-by-day simulation with:
- Starting Capital: $100,000
- Position Size: 5% of equity
- Commission: $0.005 per share (min $1)
- Long Filter: Stock must be above 150d SMA
- Ranking: By dollar volume (volume * close)
- Combined Mode: Max 50% short, 150% gross exposure cap
- 200d lookback for channel signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from strategy import get_combined_signal, load_stock_data
except ModuleNotFoundError:
    from tom_basso_channels.strategy import get_combined_signal, load_stock_data


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STARTING_CAPITAL = 100_000
POSITION_SIZE_PCT = 0.05  # 5% of equity per position
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0
LOOKBACK = 200
SMA_FILTER_PERIOD = 150
TOP_N_STOCKS = 100  # Universe filter

# Combined mode exposure limits
MAX_SHORT_EXPOSURE = 0.50  # 50% max short
MAX_GROSS_EXPOSURE = 1.50  # 150% gross exposure cap


def calculate_commission(shares: float, price: float) -> float:
    """Calculate commission for a trade."""
    commission = abs(shares) * COMMISSION_PER_SHARE
    return max(commission, MIN_COMMISSION)


def filter_top_traded_stocks(df: pd.DataFrame, top_n: int = 100) -> set:
    """Get top N most traded stocks by average dollar volume."""
    df['dollar_volume'] = df['Volume'] * df['Close']
    avg_dollar_vol = df.groupby('Symbol')['dollar_volume'].mean().sort_values(ascending=False)
    return set(avg_dollar_vol.head(top_n).index)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with signals and indicators."""
    print("Preparing data...")
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Calculate 150d SMA for long filter
    df['SMA_150'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.rolling(SMA_FILTER_PERIOD).mean()
    )
    
    # Calculate 200d channel signals
    print("Calculating channel signals...")
    all_signals = []
    for sym, grp in tqdm(df.groupby('Symbol'), desc="Computing signals"):
        signal = get_combined_signal(grp['Close'], grp['High'], grp['Low'], LOOKBACK)
        all_signals.append(pd.DataFrame({
            'idx': grp.index,
            'signal': signal.values
        }))
    
    signals_df = pd.concat(all_signals, ignore_index=True)
    df['signal'] = signals_df.set_index('idx')['signal']
    
    # Calculate dollar volume for ranking
    df['dollar_volume'] = df['Volume'] * df['Close']
    
    return df


def run_portfolio_backtest(df: pd.DataFrame, mode: str = 'long') -> dict:
    """
    Run day-by-day portfolio backtest.
    
    Args:
        df: Prepared DataFrame with signals
        mode: 'long', 'short', or 'combined'
    
    Returns:
        dict with metrics and dataframes
    """
    print(f"\nRunning {mode.upper()} backtest...")
    
    # Get unique dates
    dates = sorted(df['Date'].unique())
    
    # Initialize portfolio state
    cash = STARTING_CAPITAL
    positions = {}  # {symbol: {'shares': float, 'entry_price': float, 'entry_date': date, 'direction': 1/-1}}
    
    equity_curve = []
    trades = []
    
    for date in tqdm(dates, desc=f"{mode.upper()} simulation"):
        day_data = df[df['Date'] == date].copy()
        
        if day_data.empty:
            continue
        
        # Get current prices for existing positions
        current_prices = day_data.set_index('Symbol')['Close'].to_dict()
        current_opens = day_data.set_index('Symbol')['Open'].to_dict()
        
        # Calculate current portfolio value
        position_value = 0
        long_exposure = 0
        short_exposure = 0
        
        for sym, pos in positions.items():
            if sym in current_prices:
                price = current_prices[sym]
                value = pos['shares'] * price * pos['direction']
                position_value += value
                if pos['direction'] == 1:
                    long_exposure += abs(value)
                else:
                    short_exposure += abs(value)
        
        current_equity = cash + position_value
        
        # Process exits first
        symbols_to_close = []
        for sym, pos in positions.items():
            if sym not in current_prices:
                continue
            
            row = day_data[day_data['Symbol'] == sym].iloc[0]
            signal = row['signal']
            
            # Exit conditions
            should_exit = False
            if pos['direction'] == 1 and signal != 1:  # Long exit
                should_exit = True
            elif pos['direction'] == -1 and signal != -1:  # Short exit
                should_exit = True
            
            if should_exit:
                exit_price = current_opens[sym]  # Exit at open
                shares = pos['shares']
                direction = pos['direction']
                entry_value = pos['entry_price'] * shares
                
                # Commission
                commission = calculate_commission(shares, exit_price)
                
                # Calculate PnL and update cash
                if direction == 1:  # Long
                    exit_value = exit_price * shares
                    pnl = exit_value - entry_value - commission
                    cash += exit_value - commission
                else:  # Short: we borrowed shares at entry, now buy back
                    exit_value = exit_price * shares
                    pnl = entry_value - exit_value - commission  # Profit if price went down
                    # Return the margin + profit/loss
                    cash += pos['margin'] + pnl
                
                # Record trade
                days_held = (date - pos['entry_date']).days
                trades.append({
                    'symbol': sym,
                    'direction': 'long' if direction == 1 else 'short',
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'return_pct': pnl / (pos['entry_price'] * shares) * 100,
                    'days_held': days_held
                })
                
                symbols_to_close.append(sym)
        
        # Remove closed positions
        for sym in symbols_to_close:
            del positions[sym]
        
        # Recalculate exposures after exits
        position_value = 0
        long_exposure = 0
        short_exposure = 0
        for sym, pos in positions.items():
            if sym in current_prices:
                price = current_prices[sym]
                value = pos['shares'] * price
                position_value += value * pos['direction']
                if pos['direction'] == 1:
                    long_exposure += value
                else:
                    short_exposure += value
        
        current_equity = cash + position_value
        
        # Process entries
        # Get candidates with signals
        candidates = day_data.copy()
        
        # Apply mode filter
        if mode == 'long':
            # Only long signals, must be above 150d SMA
            candidates = candidates[(candidates['signal'] == 1) & (candidates['Close'] > candidates['SMA_150'])]
        elif mode == 'short':
            # Only short signals
            candidates = candidates[candidates['signal'] == -1]
        else:  # combined
            # Long: signal=1 and above SMA, Short: signal=-1
            long_cands = candidates[(candidates['signal'] == 1) & (candidates['Close'] > candidates['SMA_150'])].copy()
            long_cands['direction'] = 1
            short_cands = candidates[candidates['signal'] == -1].copy()
            short_cands['direction'] = -1
            candidates = pd.concat([long_cands, short_cands], ignore_index=True)
        
        # Remove symbols already in portfolio
        candidates = candidates[~candidates['Symbol'].isin(positions.keys())]
        
        # Rank by dollar volume
        candidates = candidates.sort_values('dollar_volume', ascending=False)
        
        # Calculate target position size
        target_position_value = current_equity * POSITION_SIZE_PCT
        
        for _, row in candidates.iterrows():
            sym = row['Symbol']
            price = row['Open']  # Enter at open
            
            if price <= 0 or pd.isna(price):
                continue
            
            # Determine direction
            if mode == 'long':
                direction = 1
            elif mode == 'short':
                direction = -1
            else:
                direction = row['direction']
            
            # Check exposure limits for combined mode
            if mode == 'combined':
                gross_exposure = (long_exposure + short_exposure) / current_equity
                short_pct = short_exposure / current_equity
                
                if direction == -1:
                    # Check short limit
                    if short_pct >= MAX_SHORT_EXPOSURE:
                        continue
                    # Check gross limit
                    if gross_exposure >= MAX_GROSS_EXPOSURE:
                        continue
                else:
                    # Check gross limit for longs too
                    if gross_exposure >= MAX_GROSS_EXPOSURE:
                        continue
            
            # Calculate shares
            shares = int(target_position_value / price)
            if shares <= 0:
                continue
            
            actual_value = shares * price
            commission = calculate_commission(shares, price)
            
            # Check if we have enough cash
            if direction == 1:  # Long
                required_cash = actual_value + commission
            else:  # Short: need margin (100% of position value)
                required_cash = actual_value + commission
            
            if cash < required_cash:
                continue
            
            # Execute entry
            cash -= (actual_value + commission)  # Same for both long and short (margin)
            
            positions[sym] = {
                'shares': shares,
                'entry_price': price,
                'entry_date': date,
                'direction': direction,
                'margin': actual_value if direction == -1 else 0  # Margin for shorts
            }
            
            # Update exposures
            if direction == 1:
                long_exposure += actual_value
            else:
                short_exposure += actual_value
        
        # Record equity
        position_value = sum(
            pos['shares'] * current_prices.get(sym, pos['entry_price']) * pos['direction']
            for sym, pos in positions.items()
        )
        current_equity = cash + position_value
        
        equity_curve.append({
            'Date': date,
            'Equity': current_equity,
            'Cash': cash,
            'Long_Exposure': long_exposure,
            'Short_Exposure': short_exposure,
            'N_Positions': len(positions)
        })
    
    # Close remaining positions at end
    if positions:
        last_date = dates[-1]
        last_day = df[df['Date'] == last_date]
        last_prices = last_day.set_index('Symbol')['Close'].to_dict()
        
        for sym, pos in positions.items():
            if sym in last_prices:
                exit_price = last_prices[sym]
                shares = pos['shares']
                direction = pos['direction']
                
                entry_value = pos['entry_price'] * shares
                exit_value = exit_price * shares
                commission = calculate_commission(shares, exit_price)
                
                if direction == 1:
                    pnl = exit_value - entry_value - commission
                else:
                    pnl = entry_value - exit_value - commission
                
                days_held = (last_date - pos['entry_date']).days
                trades.append({
                    'symbol': sym,
                    'direction': 'long' if direction == 1 else 'short',
                    'entry_date': pos['entry_date'],
                    'exit_date': last_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'return_pct': pnl / (pos['entry_price'] * shares) * 100,
                    'days_held': days_held
                })
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    metrics = calculate_metrics(equity_df, trades_df, mode)
    
    return {
        'metrics': metrics,
        'equity_df': equity_df,
        'trades_df': trades_df
    }


def calculate_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, mode: str) -> dict:
    """Calculate comprehensive backtest metrics."""
    
    if equity_df.empty:
        return {'mode': mode, 'error': 'No data'}
    
    # Basic metrics
    start_equity = STARTING_CAPITAL
    final_equity = equity_df.iloc[-1]['Equity']
    total_return = (final_equity - start_equity) / start_equity * 100
    
    # CAGR
    start_date = equity_df.iloc[0]['Date']
    end_date = equity_df.iloc[-1]['Date']
    n_years = (end_date - start_date).days / 365.25
    if n_years > 0 and final_equity > 0:
        cagr = ((final_equity / start_equity) ** (1 / n_years) - 1) * 100
    else:
        cagr = 0
    
    # Daily returns for Sharpe
    equity_df['Daily_Return'] = equity_df['Equity'].pct_change()
    daily_returns = equity_df['Daily_Return'].dropna()
    
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max Drawdown
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_dd = equity_df['Drawdown'].min() * 100
    
    # Trade metrics
    if not trades_df.empty:
        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
        win_pct = win_rate
        loss_pct = 100 - win_rate
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        avg_days_held = trades_df['days_held'].mean()
        avg_win = winning_trades['return_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['return_pct'].mean() if len(losing_trades) > 0 else 0
    else:
        n_trades = 0
        win_rate = win_pct = loss_pct = 0
        profit_factor = 0
        avg_days_held = 0
        avg_win = avg_loss = 0
    
    return {
        'mode': mode,
        'start_date': start_date,
        'end_date': end_date,
        'n_years': n_years,
        'start_equity': start_equity,
        'final_equity': final_equity,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'win_pct': win_pct,
        'loss_pct': loss_pct,
        'profit_factor': profit_factor,
        'avg_days_held': avg_days_held,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss
    }


def print_results(metrics: dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"  {metrics['mode'].upper()} STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"  Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')} ({metrics['n_years']:.1f} years)")
    print(f"  Starting Capital: ${metrics['start_equity']:,.0f}")
    print(f"  Final Equity: ${metrics['final_equity']:,.0f}")
    print()
    print(f"  {'Total Return:':<20} {metrics['total_return']:>10.2f}%")
    print(f"  {'CAGR:':<20} {metrics['cagr']:>10.2f}%")
    print(f"  {'Sharpe Ratio:':<20} {metrics['sharpe']:>10.4f}")
    print(f"  {'Profit Factor:':<20} {metrics['profit_factor']:>10.4f}")
    print(f"  {'Max Drawdown:':<20} {metrics['max_dd']:>10.2f}%")
    print()
    print(f"  {'Win Rate:':<20} {metrics['win_rate']:>10.2f}%")
    print(f"  {'Win %:':<20} {metrics['win_pct']:>10.2f}%")
    print(f"  {'Loss %:':<20} {metrics['loss_pct']:>10.2f}%")
    print(f"  {'Avg Days Held:':<20} {metrics['avg_days_held']:>10.1f}")
    print(f"  {'N Trades:':<20} {metrics['n_trades']:>10.0f}")
    print(f"  {'Avg Win:':<20} {metrics['avg_win_pct']:>10.2f}%")
    print(f"  {'Avg Loss:':<20} {metrics['avg_loss_pct']:>10.2f}%")


def main():
    print("=" * 70)
    print("TOM BASSO CHANNEL STRATEGY - PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"Starting Capital: ${STARTING_CAPITAL:,}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}% of equity")
    print(f"Lookback: {LOOKBACK} days")
    print(f"SMA Filter: {SMA_FILTER_PERIOD} days (long only)")
    print(f"Universe: Top {TOP_N_STOCKS} stocks by dollar volume")
    print(f"Commission: ${COMMISSION_PER_SHARE}/share (min ${MIN_COMMISSION})")
    print("=" * 70)
    
    # Load data
    print("\nLoading stock data...")
    df = load_stock_data()
    df = df[df['Close'] > 0].copy()
    print(f"Loaded {df['Symbol'].nunique()} symbols")
    
    # Filter to top 100 most traded stocks
    print("Filtering to top 100 most traded stocks...")
    top_symbols = filter_top_traded_stocks(df, top_n=TOP_N_STOCKS)
    df = df[df['Symbol'].isin(top_symbols)].copy()
    print(f"Using {len(top_symbols)} symbols")
    
    # Prepare data
    df = prepare_data(df)
    
    # Run backtests
    results = {}
    
    for mode in ['long', 'short', 'combined']:
        result = run_portfolio_backtest(df.copy(), mode=mode)
        results[mode] = result
        print_results(result['metrics'])
    
    # Summary comparison
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<20} {'LONG':>18} {'SHORT':>18} {'COMBINED':>18}")
    print("-" * 90)
    
    metrics_to_show = [
        ('Final Equity', 'final_equity', '${:,.0f}'),
        ('Total Return %', 'total_return', '{:.2f}%'),
        ('CAGR %', 'cagr', '{:.2f}%'),
        ('Sharpe Ratio', 'sharpe', '{:.4f}'),
        ('Profit Factor', 'profit_factor', '{:.4f}'),
        ('Max Drawdown %', 'max_dd', '{:.2f}%'),
        ('Win Rate %', 'win_rate', '{:.2f}%'),
        ('Avg Days Held', 'avg_days_held', '{:.1f}'),
        ('N Trades', 'n_trades', '{:.0f}'),
    ]
    
    for label, key, fmt in metrics_to_show:
        long_val = fmt.format(results['long']['metrics'][key])
        short_val = fmt.format(results['short']['metrics'][key])
        combined_val = fmt.format(results['combined']['metrics'][key])
        print(f"{label:<20} {long_val:>18} {short_val:>18} {combined_val:>18}")
    
    print("=" * 90)
    
    # Plot equity curves
    output_dir = Path(__file__).parent
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    colors = {'long': '#00ff00', 'short': '#ff4444', 'combined': '#00ccff'}
    
    for mode, result in results.items():
        eq_df = result['equity_df']
        m = result['metrics']
        label = f"{mode.upper()} (CAGR: {m['cagr']:.1f}%, Sharpe: {m['sharpe']:.2f})"
        plt.plot(eq_df['Date'], eq_df['Equity'], color=colors[mode], linewidth=1.5, label=label)
    
    plt.axhline(y=STARTING_CAPITAL, color='white', linestyle='--', alpha=0.5, label='Starting Capital')
    plt.title('Tom Basso Channel Strategy - Equity Curves\n(200d Lookback, Top 100 Stocks, 5% Position Size)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / "backtest_equity_curves.png"
    plt.savefig(plot_path, facecolor='black', dpi=150)
    print(f"\nEquity curves saved to {plot_path}")
    plt.close()
    
    # Save results
    results_data = [results['long']['metrics'], results['short']['metrics'], results['combined']['metrics']]
    results_df = pd.DataFrame(results_data)
    results_path = output_dir / "backtest_portfolio_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
