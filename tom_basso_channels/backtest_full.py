"""
Tom Basso Channel Strategy - Full Backtest

Backtest with:
- 200-day lookback for combined channel signal (Donchian + Keltner + Bollinger)
- Top 100 most traded stocks filter (volume * close) as universe filter
- Long-only, Short-only, and Combined strategies
- Comprehensive metrics: Sharpe, Win%, Loss%, Avg Days Held, Win Rate, CAGR, Total Return, PF, Max DD
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from strategy import get_combined_signal, load_stock_data
except ModuleNotFoundError:
    from tom_basso_channels.strategy import get_combined_signal, load_stock_data


def filter_top_traded_stocks(df: pd.DataFrame, top_n: int = 100) -> set:
    """
    Get the set of top N most traded stocks by dollar volume (volume * close).
    This is calculated once at the start to define the tradeable universe.
    """
    # Calculate average dollar volume per symbol
    df['dollar_volume'] = df['Volume'] * df['Close']
    avg_dollar_vol = df.groupby('Symbol')['dollar_volume'].mean().sort_values(ascending=False)
    top_symbols = set(avg_dollar_vol.head(top_n).index)
    return top_symbols


def run_backtest(df: pd.DataFrame, lookback: int = 200, mode: str = 'combined') -> dict:
    """
    Run backtest for a given mode.
    
    Args:
        df: DataFrame with OHLCV data for filtered universe
        lookback: Lookback period for channel indicators
        mode: 'long', 'short', or 'combined'
    
    Returns:
        dict with all metrics and trade details
    """
    # Sort by symbol and date
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Calculate signals per symbol
    all_results = []
    
    for sym, grp in df.groupby('Symbol'):
        grp = grp.copy()
        signal = get_combined_signal(grp['Close'], grp['High'], grp['Low'], lookback)
        
        # Apply mode filter
        if mode == 'long':
            signal = signal.clip(lower=0)  # Only long positions (1 or 0)
        elif mode == 'short':
            signal = signal.clip(upper=0)  # Only short positions (-1 or 0)
        # 'combined' uses full signal (-1, 0, 1)
        
        grp['signal'] = signal.values
        grp['log_return'] = np.log(grp['Close']).diff().shift(-1)
        grp['strategy_return'] = grp['signal'] * grp['log_return']
        
        all_results.append(grp)
    
    result_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate mean returns per date (equal-weighted portfolio)
    daily_returns = result_df.groupby('Date').agg({
        'strategy_return': 'mean',
        'signal': lambda x: (x != 0).sum()  # Number of positions
    }).rename(columns={'signal': 'n_positions'})
    
    daily_returns = daily_returns.sort_index()
    daily_returns = daily_returns.dropna()
    
    # Calculate metrics
    metrics = calculate_metrics(daily_returns, result_df, mode)
    
    return metrics


def calculate_metrics(daily_returns: pd.DataFrame, result_df: pd.DataFrame, mode: str) -> dict:
    """Calculate comprehensive backtest metrics."""
    
    returns = daily_returns['strategy_return']
    
    # Basic return metrics
    total_log_return = returns.sum()
    total_return = np.exp(total_log_return) - 1
    
    # CAGR
    n_days = len(returns)
    n_years = n_days / 252
    if n_years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / n_years) - 1
    else:
        cagr = np.nan
    
    # Sharpe Ratio (annualized)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = np.nan
    
    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    profit_factor = gains / losses if losses > 0 else np.inf
    
    # Max Drawdown
    cumulative = returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()
    
    # Win/Loss metrics (based on daily returns)
    n_winning_days = (returns > 0).sum()
    n_losing_days = (returns < 0).sum()
    n_total_days = n_winning_days + n_losing_days
    
    win_rate = n_winning_days / n_total_days if n_total_days > 0 else np.nan
    win_pct = win_rate * 100
    loss_pct = (1 - win_rate) * 100 if not np.isnan(win_rate) else np.nan
    
    # Trade-level analysis (for average days held)
    trade_stats = analyze_trades(result_df, mode)
    
    return {
        'mode': mode,
        'lookback': 200,
        'n_symbols': result_df['Symbol'].nunique(),
        'total_return': total_return * 100,  # as percentage
        'cagr': cagr * 100,  # as percentage
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown * 100,  # as percentage (log return)
        'win_rate': win_rate * 100,  # as percentage
        'win_pct': win_pct,
        'loss_pct': loss_pct,
        'avg_days_held': trade_stats['avg_days_held'],
        'n_trades': trade_stats['n_trades'],
        'n_trading_days': n_days,
        'start_date': daily_returns.index.min(),
        'end_date': daily_returns.index.max(),
    }


def analyze_trades(result_df: pd.DataFrame, mode: str) -> dict:
    """Analyze individual trades to get average holding period."""
    
    all_trades = []
    
    for sym, grp in result_df.groupby('Symbol'):
        grp = grp.sort_values('Date').reset_index(drop=True)
        
        # Detect trade entries and exits
        signal = grp['signal'].values
        dates = grp['Date'].values
        
        in_trade = False
        entry_idx = None
        entry_signal = 0
        
        for i in range(len(signal)):
            if not in_trade:
                # Check for entry
                if mode == 'long' and signal[i] == 1:
                    in_trade = True
                    entry_idx = i
                    entry_signal = 1
                elif mode == 'short' and signal[i] == -1:
                    in_trade = True
                    entry_idx = i
                    entry_signal = -1
                elif mode == 'combined' and signal[i] != 0:
                    in_trade = True
                    entry_idx = i
                    entry_signal = signal[i]
            else:
                # Check for exit (signal changes or goes to 0)
                if signal[i] != entry_signal:
                    # Trade closed
                    days_held = i - entry_idx
                    all_trades.append({
                        'symbol': sym,
                        'entry_date': dates[entry_idx],
                        'exit_date': dates[i],
                        'days_held': days_held,
                        'direction': 'long' if entry_signal == 1 else 'short'
                    })
                    in_trade = False
                    
                    # Check if new trade starts immediately
                    if mode == 'long' and signal[i] == 1:
                        in_trade = True
                        entry_idx = i
                        entry_signal = 1
                    elif mode == 'short' and signal[i] == -1:
                        in_trade = True
                        entry_idx = i
                        entry_signal = -1
                    elif mode == 'combined' and signal[i] != 0:
                        in_trade = True
                        entry_idx = i
                        entry_signal = signal[i]
    
    if len(all_trades) == 0:
        return {'avg_days_held': np.nan, 'n_trades': 0}
    
    trades_df = pd.DataFrame(all_trades)
    avg_days_held = trades_df['days_held'].mean()
    
    return {
        'avg_days_held': avg_days_held,
        'n_trades': len(trades_df)
    }


def print_results(metrics: dict):
    """Print formatted results for a single backtest."""
    print(f"\n{'='*60}")
    print(f"  {metrics['mode'].upper()} STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"  Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}")
    print(f"  Symbols: {metrics['n_symbols']}")
    print(f"  Trading Days: {metrics['n_trading_days']}")
    print(f"  Number of Trades: {metrics['n_trades']}")
    print()
    print(f"  {'Total Return:':<20} {metrics['total_return']:>10.2f}%")
    print(f"  {'CAGR:':<20} {metrics['cagr']:>10.2f}%")
    print(f"  {'Sharpe Ratio:':<20} {metrics['sharpe_ratio']:>10.4f}")
    print(f"  {'Profit Factor:':<20} {metrics['profit_factor']:>10.4f}")
    print(f"  {'Max Drawdown:':<20} {metrics['max_drawdown']:>10.2f}%")
    print()
    print(f"  {'Win Rate:':<20} {metrics['win_rate']:>10.2f}%")
    print(f"  {'Win %:':<20} {metrics['win_pct']:>10.2f}%")
    print(f"  {'Loss %:':<20} {metrics['loss_pct']:>10.2f}%")
    print(f"  {'Avg Days Held:':<20} {metrics['avg_days_held']:>10.1f}")


def main():
    print("=" * 60)
    print("TOM BASSO CHANNEL STRATEGY - FULL BACKTEST")
    print("Lookback: 200 days | Universe: Top 100 Most Traded Stocks")
    print("=" * 60)
    
    # Load data
    print("\nLoading stock data...")
    df = load_stock_data()
    df = df[df['Close'] > 0].copy()
    print(f"Loaded {df['Symbol'].nunique()} symbols")
    
    # Filter to top 100 most traded stocks
    print("Filtering to top 100 most traded stocks...")
    top_symbols = filter_top_traded_stocks(df, top_n=100)
    df = df[df['Symbol'].isin(top_symbols)].copy()
    print(f"Using {len(top_symbols)} symbols")
    
    # Run backtests
    results = {}
    
    for mode in ['long', 'short', 'combined']:
        print(f"\nRunning {mode.upper()} backtest...")
        metrics = run_backtest(df.copy(), lookback=200, mode=mode)
        results[mode] = metrics
        print_results(metrics)
    
    # Summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'LONG':>15} {'SHORT':>15} {'COMBINED':>15}")
    print("-" * 80)
    
    metrics_to_show = [
        ('Total Return %', 'total_return'),
        ('CAGR %', 'cagr'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Profit Factor', 'profit_factor'),
        ('Max Drawdown %', 'max_drawdown'),
        ('Win Rate %', 'win_rate'),
        ('Avg Days Held', 'avg_days_held'),
        ('N Trades', 'n_trades'),
    ]
    
    for label, key in metrics_to_show:
        long_val = results['long'][key]
        short_val = results['short'][key]
        combined_val = results['combined'][key]
        
        if key in ['n_trades']:
            print(f"{label:<20} {long_val:>15.0f} {short_val:>15.0f} {combined_val:>15.0f}")
        else:
            print(f"{label:<20} {long_val:>15.2f} {short_val:>15.2f} {combined_val:>15.2f}")
    
    print("=" * 80)
    
    # Save results to CSV
    output_dir = Path(__file__).parent
    results_df = pd.DataFrame([results['long'], results['short'], results['combined']])
    results_path = output_dir / "backtest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
