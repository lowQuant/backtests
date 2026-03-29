"""
Profit Factor vs Lookback Analysis for Tom Basso Channel Strategy

Generates a plot showing profit factor on the y-axis and lookback values on the x-axis
for the combined Donchian + Keltner + Bollinger signal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

try:
    from strategy import (get_combined_signal, load_stock_data, calculate_metrics, 
                          run_universe_backtest_vectorized, calculate_profit_factor_from_mean_returns)
except ModuleNotFoundError:
    from tom_basso_channels.strategy import (get_combined_signal, load_stock_data, calculate_metrics, 
                                              run_universe_backtest_vectorized, calculate_profit_factor_from_mean_returns)


def analyze_lookback_range(min_lookback: int = 5, 
                           max_lookback: int = 255, 
                           step: int = 5,
                           data_path: Path = None,
                           max_symbols: int = None) -> pd.DataFrame:
    """
    Analyze profit factor across a range of lookback values using vectorized operations.
    
    Returns a DataFrame with lookback, profit_factor, sharpe_ratio columns.
    Uses mean returns across symbols by date for proper cross-sectional analysis.
    """
    print("Loading stock data...")
    df = load_stock_data(data_path)
    
    # Filter invalid prices
    df = df[df['Close'] > 0].copy()
    
    symbols = df['Symbol'].unique()
    print(f"Loaded {len(symbols)} symbols")
    
    # Optionally limit symbols for faster analysis
    if max_symbols is not None and len(symbols) > max_symbols:
        avg_volume = df.groupby('Symbol')['Volume'].mean().sort_values(ascending=False)
        top_symbols = avg_volume.head(max_symbols).index.tolist()
        df = df[df['Symbol'].isin(top_symbols)]
        print(f"Using top {len(top_symbols)} symbols by volume")
    
    # Filter symbols with sufficient data for max lookback
    symbol_counts = df.groupby('Symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= max_lookback * 2].index
    df = df[df['Symbol'].isin(valid_symbols)]
    print(f"Using {len(valid_symbols)} symbols with sufficient data")
    
    # Sort once for all operations and pre-compute log returns
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    df['log_return'] = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x).diff().shift(-1))
    
    results = []
    lookbacks = range(min_lookback, max_lookback + 1, step)
    
    # Pre-group data by symbol for faster iteration
    symbol_groups = [(sym, grp.copy()) for sym, grp in df.groupby('Symbol')]
    print(f"Processing {len(symbol_groups)} symbols...")
    
    for lookback in tqdm(lookbacks, desc="Analyzing lookbacks"):
        # Collect strategy returns with dates for each symbol
        all_strat_returns = []
        
        for sym, grp in symbol_groups:
            signal = get_combined_signal(grp['Close'], grp['High'], grp['Low'], lookback)
            strat_ret = signal * grp['log_return']
            sym_df = pd.DataFrame({'Date': grp['Date'].values, 'strategy_return': strat_ret.values})
            all_strat_returns.append(sym_df)
        
        # Combine and calculate mean returns per date
        combined = pd.concat(all_strat_returns, ignore_index=True)
        mean_returns = combined.groupby('Date')['strategy_return'].mean().dropna()
        
        # Calculate profit factor from mean returns
        gains = mean_returns[mean_returns > 0].sum()
        losses = mean_returns[mean_returns < 0].abs().sum()
        profit_factor = gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)
        
        # Calculate other metrics from mean returns
        metrics = calculate_metrics(mean_returns)
        
        results.append({
            'lookback': lookback,
            'profit_factor': profit_factor,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_return': metrics['total_return']
        })
    
    return pd.DataFrame(results)


def plot_profit_factor_curve(results_df: pd.DataFrame, 
                             output_path: Path = None,
                             show: bool = True):
    """
    Plot profit factor vs lookback with Bloomberg-style dark theme.
    """
    plt.style.use('dark_background')
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Profit Factor (primary y-axis)
    color1 = '#00ff00'  # Green
    ax1.plot(results_df['lookback'], results_df['profit_factor'], 
             color=color1, linewidth=2.5, marker='o', markersize=6, label='Profit Factor')
    ax1.set_xlabel('Lookback Period (days)', fontsize=14, color='white')
    ax1.set_ylabel('Profit Factor', fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=1.0, color='#ff0000', linestyle='--', linewidth=1, alpha=0.7, label='Break-even (PF=1)')
    
    # Sharpe Ratio (secondary y-axis)
    ax2 = ax1.twinx()
    color2 = '#00ccff'  # Cyan
    ax2.plot(results_df['lookback'], results_df['sharpe_ratio'], 
             color=color2, linewidth=2, linestyle='--', marker='s', markersize=5, label='Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio (Annualized)', fontsize=14, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and grid
    ax1.set_title('Tom Basso Channel Strategy: Profit Factor vs Lookback\n'
                  '(Donchian + Keltner + Bollinger Combined Signal)', 
                  fontsize=16, color='white', fontweight='bold', pad=20)
    ax1.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               facecolor='black', edgecolor='white', fontsize=11)
    
    # Highlight best lookback
    best_idx = results_df['profit_factor'].idxmax()
    best_lookback = results_df.loc[best_idx, 'lookback']
    best_pf = results_df.loc[best_idx, 'profit_factor']
    ax1.scatter([best_lookback], [best_pf], color='#ffff00', s=200, zorder=5, 
                edgecolors='white', linewidths=2, marker='*')
    ax1.annotate(f'Best: {best_lookback}d\nPF={best_pf:.3f}', 
                 xy=(best_lookback, best_pf), 
                 xytext=(best_lookback + 5, best_pf + 0.05),
                 fontsize=11, color='#ffff00',
                 arrowprops=dict(arrowstyle='->', color='#ffff00', lw=1.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, facecolor='black', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def print_summary(results_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("TOM BASSO CHANNEL STRATEGY - LOOKBACK ANALYSIS SUMMARY")
    print("=" * 60)
    
    best_pf_idx = results_df['profit_factor'].idxmax()
    best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
    
    print(f"\nBest Profit Factor:")
    print(f"  Lookback: {results_df.loc[best_pf_idx, 'lookback']} days")
    print(f"  Profit Factor: {results_df.loc[best_pf_idx, 'profit_factor']:.4f}")
    print(f"  Sharpe Ratio: {results_df.loc[best_pf_idx, 'sharpe_ratio']:.4f}")
    
    print(f"\nBest Sharpe Ratio:")
    print(f"  Lookback: {results_df.loc[best_sharpe_idx, 'lookback']} days")
    print(f"  Profit Factor: {results_df.loc[best_sharpe_idx, 'profit_factor']:.4f}")
    print(f"  Sharpe Ratio: {results_df.loc[best_sharpe_idx, 'sharpe_ratio']:.4f}")
    
    print(f"\nLookback Range Statistics:")
    print(f"  Mean Profit Factor: {results_df['profit_factor'].mean():.4f}")
    print(f"  Std Profit Factor: {results_df['profit_factor'].std():.4f}")
    print(f"  Mean Sharpe Ratio: {results_df['sharpe_ratio'].mean():.4f}")
    
    # Show all results
    print("\n" + "-" * 60)
    print("Full Results:")
    print("-" * 60)
    print(results_df.to_string(index=False))


def main():
    output_dir = Path(__file__).parent
    
    # Analyze lookback range
    results = analyze_lookback_range(
        min_lookback=100,
        max_lookback=250,
        step=5
    )
    
    # Save results
    results_path = output_dir / "lookback_analysis_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot
    plot_path = output_dir / "profit_factor_vs_lookback.png"
    plot_profit_factor_curve(results, output_path=plot_path, show=False)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
