"""
Tom Basso Channel Strategy - Improved Short Strategy

Short-term mean reversion approach for shorts:
- Entry: 3-day channel breakdown (close < 3d low)
- Exit: Close < 5d EMA (cover when momentum shifts)

This is designed to capture short-term overbought conditions rather than
trend-following on the short side (which fails in bull markets).
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
POSITION_SIZE_PCT = 0.05
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0

# Long strategy params (unchanged)
LONG_LOOKBACK = 200
SMA_FILTER_PERIOD = 150

# Improved short strategy params
SHORT_LOOKBACK = 3  # Short-term channel
SHORT_EXIT_EMA = 5  # Exit when close < 5d EMA

TOP_N_STOCKS = 100

# Combined mode exposure limits
MAX_SHORT_EXPOSURE = 0.50
MAX_GROSS_EXPOSURE = 1.50


def calculate_commission(shares: float, price: float) -> float:
    """Calculate commission for a trade."""
    commission = abs(shares) * COMMISSION_PER_SHARE
    return max(commission, MIN_COMMISSION)


def filter_top_traded_stocks(df: pd.DataFrame, top_n: int = 100) -> set:
    """Get top N most traded stocks by average dollar volume."""
    df['dollar_volume'] = df['Volume'] * df['Close']
    avg_dollar_vol = df.groupby('Symbol')['dollar_volume'].mean().sort_values(ascending=False)
    return set(avg_dollar_vol.head(top_n).index)


def short_signal_improved(close: pd.Series, high: pd.Series, low: pd.Series, 
                          lookback: int = 3) -> pd.Series:
    """
    Improved short signal: Short when close breaks below N-day low.
    This is a short-term overbought reversal signal.
    """
    # Entry: Close > N-day high (overbought, potential reversal)
    upper = high.rolling(lookback).max().shift(1)
    
    signal = pd.Series(np.nan, index=close.index)
    signal.loc[close > upper] = -1  # Short signal when breaking above
    signal = signal.ffill().fillna(0)
    
    return signal


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with signals and indicators."""
    print("Preparing data...")
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Calculate 150d SMA for long filter
    df['SMA_150'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.rolling(SMA_FILTER_PERIOD).mean()
    )
    
    # Calculate 5d EMA for short exit
    df['EMA_5'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.ewm(span=SHORT_EXIT_EMA, adjust=False).mean()
    )
    
    # Calculate 200d channel signals for LONG
    print("Calculating LONG signals (200d channels)...")
    long_signals = []
    for sym, grp in tqdm(df.groupby('Symbol'), desc="Long signals"):
        signal = get_combined_signal(grp['Close'], grp['High'], grp['Low'], LONG_LOOKBACK)
        # Only keep buy signals (1), ignore sell signals for long-only
        signal = signal.clip(lower=0)
        long_signals.append(pd.DataFrame({'idx': grp.index, 'long_signal': signal.values}))
    
    long_df = pd.concat(long_signals, ignore_index=True)
    df['long_signal'] = long_df.set_index('idx')['long_signal']
    
    # Calculate 3d channel signals for SHORT
    print("Calculating SHORT signals (3d breakout)...")
    short_signals = []
    for sym, grp in tqdm(df.groupby('Symbol'), desc="Short signals"):
        # Short when price breaks above 3d high (overbought)
        upper = grp['High'].rolling(SHORT_LOOKBACK).max().shift(1)
        signal = pd.Series(0, index=grp.index)
        signal.loc[grp['Close'] > upper] = -1
        short_signals.append(pd.DataFrame({'idx': grp.index, 'short_signal': signal.values}))
    
    short_df = pd.concat(short_signals, ignore_index=True)
    df['short_signal'] = short_df.set_index('idx')['short_signal']
    
    # Calculate dollar volume for ranking
    df['dollar_volume'] = df['Volume'] * df['Close']
    
    return df


def run_portfolio_backtest(df: pd.DataFrame, mode: str = 'long') -> dict:
    """
    Run day-by-day portfolio backtest with improved short logic.
    """
    print(f"\nRunning {mode.upper()} backtest...")
    
    dates = sorted(df['Date'].unique())
    
    cash = STARTING_CAPITAL
    positions = {}
    
    equity_curve = []
    trades = []
    
    for date in tqdm(dates, desc=f"{mode.upper()} simulation"):
        day_data = df[df['Date'] == date].copy()
        
        if day_data.empty:
            continue
        
        current_prices = day_data.set_index('Symbol')['Close'].to_dict()
        current_opens = day_data.set_index('Symbol')['Open'].to_dict()
        current_ema5 = day_data.set_index('Symbol')['EMA_5'].to_dict()
        
        # Calculate current portfolio value
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
        
        # Process exits
        symbols_to_close = []
        for sym, pos in positions.items():
            if sym not in current_prices:
                continue
            
            row = day_data[day_data['Symbol'] == sym].iloc[0]
            
            should_exit = False
            
            if pos['direction'] == 1:  # Long exit
                long_sig = row['long_signal']
                if long_sig != 1:
                    should_exit = True
            else:  # Short exit - improved: exit when close < 5d EMA
                close = current_prices[sym]
                ema5 = current_ema5.get(sym, close)
                if close < ema5:  # Price dropped below EMA - cover short
                    should_exit = True
            
            if should_exit:
                exit_price = current_opens[sym]
                shares = pos['shares']
                direction = pos['direction']
                entry_value = pos['entry_price'] * shares
                
                commission = calculate_commission(shares, exit_price)
                
                if direction == 1:
                    exit_value = exit_price * shares
                    pnl = exit_value - entry_value - commission
                    cash += exit_value - commission
                else:
                    exit_value = exit_price * shares
                    pnl = entry_value - exit_value - commission
                    cash += pos['margin'] + pnl
                
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
                    'return_pct': pnl / entry_value * 100,
                    'days_held': days_held
                })
                
                symbols_to_close.append(sym)
        
        for sym in symbols_to_close:
            del positions[sym]
        
        # Recalculate exposures
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
        candidates = day_data.copy()
        
        if mode == 'long':
            candidates = candidates[
                (candidates['long_signal'] == 1) & 
                (candidates['Close'] > candidates['SMA_150'])
            ]
            candidates['direction'] = 1
        elif mode == 'short':
            candidates = candidates[candidates['short_signal'] == -1]
            candidates['direction'] = -1
        else:  # combined
            long_cands = candidates[
                (candidates['long_signal'] == 1) & 
                (candidates['Close'] > candidates['SMA_150'])
            ].copy()
            long_cands['direction'] = 1
            
            short_cands = candidates[candidates['short_signal'] == -1].copy()
            short_cands['direction'] = -1
            
            candidates = pd.concat([long_cands, short_cands], ignore_index=True)
        
        candidates = candidates[~candidates['Symbol'].isin(positions.keys())]
        candidates = candidates.sort_values('dollar_volume', ascending=False)
        
        target_position_value = current_equity * POSITION_SIZE_PCT
        
        for _, row in candidates.iterrows():
            sym = row['Symbol']
            price = row['Open']
            
            if price <= 0 or pd.isna(price):
                continue
            
            direction = int(row['direction'])
            
            # Check exposure limits for combined mode
            if mode == 'combined':
                gross_exposure = (long_exposure + short_exposure) / current_equity if current_equity > 0 else 0
                short_pct = short_exposure / current_equity if current_equity > 0 else 0
                
                if direction == -1:
                    if short_pct >= MAX_SHORT_EXPOSURE:
                        continue
                    if gross_exposure >= MAX_GROSS_EXPOSURE:
                        continue
                else:
                    if gross_exposure >= MAX_GROSS_EXPOSURE:
                        continue
            
            shares = int(target_position_value / price)
            if shares <= 0:
                continue
            
            actual_value = shares * price
            commission = calculate_commission(shares, price)
            required_cash = actual_value + commission
            
            if cash < required_cash:
                continue
            
            cash -= required_cash
            
            positions[sym] = {
                'shares': shares,
                'entry_price': price,
                'entry_date': date,
                'direction': direction,
                'margin': actual_value if direction == -1 else 0
            }
            
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
    
    # Close remaining positions
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
                    'return_pct': pnl / entry_value * 100,
                    'days_held': days_held
                })
    
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
    
    start_equity = STARTING_CAPITAL
    final_equity = equity_df.iloc[-1]['Equity']
    total_return = (final_equity - start_equity) / start_equity * 100
    
    start_date = equity_df.iloc[0]['Date']
    end_date = equity_df.iloc[-1]['Date']
    n_years = (end_date - start_date).days / 365.25
    
    if n_years > 0 and final_equity > 0:
        cagr = ((final_equity / start_equity) ** (1 / n_years) - 1) * 100
    else:
        cagr = 0
    
    equity_df['Daily_Return'] = equity_df['Equity'].pct_change()
    daily_returns = equity_df['Daily_Return'].dropna()
    
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_dd = equity_df['Drawdown'].min() * 100
    
    if not trades_df.empty:
        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        avg_days_held = trades_df['days_held'].mean()
        avg_win = winning_trades['return_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['return_pct'].mean() if len(losing_trades) > 0 else 0
    else:
        n_trades = win_rate = profit_factor = avg_days_held = avg_win = avg_loss = 0
    
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
    print(f"  {'Avg Days Held:':<20} {metrics['avg_days_held']:>10.1f}")
    print(f"  {'N Trades:':<20} {metrics['n_trades']:>10.0f}")
    print(f"  {'Avg Win:':<20} {metrics['avg_win_pct']:>10.2f}%")
    print(f"  {'Avg Loss:':<20} {metrics['avg_loss_pct']:>10.2f}%")


def main():
    print("=" * 70)
    print("TOM BASSO CHANNEL STRATEGY - IMPROVED SHORT BACKTEST")
    print("=" * 70)
    print(f"Starting Capital: ${STARTING_CAPITAL:,}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}% of equity")
    print(f"LONG: {LONG_LOOKBACK}d channels, 150d SMA filter")
    print(f"SHORT: {SHORT_LOOKBACK}d breakout entry, {SHORT_EXIT_EMA}d EMA exit")
    print(f"Universe: Top {TOP_N_STOCKS} stocks by dollar volume")
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
    print("SUMMARY COMPARISON (IMPROVED SHORT)")
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
    plt.title(f'Tom Basso Channel Strategy - IMPROVED SHORT\n(Long: 200d, Short: {SHORT_LOOKBACK}d entry + {SHORT_EXIT_EMA}d EMA exit)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / "backtest_improved_short.png"
    plt.savefig(plot_path, facecolor='black', dpi=150)
    print(f"\nEquity curves saved to {plot_path}")
    plt.close()
    
    # Save results
    results_data = [results['long']['metrics'], results['short']['metrics'], results['combined']['metrics']]
    results_df = pd.DataFrame(results_data)
    results_path = output_dir / "backtest_improved_short_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
