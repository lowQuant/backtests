import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_clean_factor_and_forward_returns(factor, prices, periods=(1, 5, 10), quantiles=5, filter_zscore=20):
    """
    Formats the factor data and pricing data into a unified DataFrame aligned on (date, asset).
    Mimics alphalens.utils.get_clean_factor_and_forward_returns
    
    Args:
    - factor: pd.Series or pd.DataFrame - MultiIndex (date, asset) containing factor values.
    - prices: pd.DataFrame - Index (date), Columns (asset) containing close prices.
    - periods: list of integers - lookahead periods for forward returns.
    - quantiles: int - number of buckets to split the factor into.
    
    Returns:
    - merged_data: pd.DataFrame with MultiIndex (date, asset), columns [factor, period_1D, period_5D, ..., quantile]
    """
    # 1. Compute Forward Returns
    forward_returns = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    period_cols = {}
    for p in periods:
        # Return = (Price_t+p - Price_t) / Price_t
        # Shifted back to time t
        ret = prices.pct_change(p).shift(-p)
        
        # Stack to (date, asset)
        stack_ret = ret.stack()
        stack_ret.name = f'{p}D'
        period_cols[f'{p}D'] = stack_ret

    # 2. Prepare Factor
    if isinstance(factor, pd.DataFrame):
        if len(factor.columns) == 1:
            factor = factor.iloc[:, 0]
        else:
            # Assume it's already properly formatted or take first col
            factor = factor.iloc[:, 0]
    
    factor.name = 'factor'
    
    # 3. Merge
    # Standardize index names to ensure join works
    if factor.index.nlevels == 2:
        factor.index.names = ['date', 'asset']
    
    merged = pd.DataFrame(factor).copy()
    for p, ret_series in period_cols.items():
        # Ensure return series has same index names
        if ret_series.index.nlevels == 2:
            ret_series.index.names = ['date', 'asset']
            
        merged = merged.join(ret_series, how='left')
    
    # Filter NaNs in Factor or Returns (Alphalens drops rows where ANY return is missing?)
    # We will be lenient and allow some missing returns if we have at least one?
    # No, for comparable analysis, usually we want valid data.
    # But for 5D returns we lose the last 5 days.
    merged = merged.dropna()
    
    # 4. Quantiles
    # Compute quantiles per date (cross-sectional)
    def quantile_calc(x):
        try:
            return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
        except ValueError:
            # Not enough unique values
            return np.nan

    merged['factor_quantile'] = merged.groupby(level=0)['factor'].transform(quantile_calc)
    
    merged = merged.dropna(subset=['factor_quantile'])
    merged['factor_quantile'] = merged['factor_quantile'].astype(int)
    
    return merged

def create_performance_tearsheet(merged_data, factor_name="IV"):
    """
    Generates tables and plots.
    """
    
    # 1. Mean Return by Quantile (Table)
    periods = [c for c in merged_data.columns if c.endswith('D')]
    
    mean_ret = merged_data.groupby('factor_quantile')[periods].mean()
    std_error = merged_data.groupby('factor_quantile')[periods].sem()
    
    print("\n" + "="*50)
    print(f"MEAN PERIOD RETURN BY {factor_name} QUANTILE (bps)")
    print("="*50)
    print((mean_ret * 10000).round(2))
    
    # 2. Spread Analysis (Q5 - Q1)
    # Assuming Quantiles are 1-based (1 to 5)
    max_q = merged_data['factor_quantile'].max()
    min_q = merged_data['factor_quantile'].min()
    
    spread = mean_ret.loc[max_q] - mean_ret.loc[min_q]
    
    print("\n" + "="*50)
    print(f"SPREAD (Q{max_q} - Q{min_q}) (bps)")
    print("="*50)
    print((spread * 10000).round(2))
    
    return mean_ret, spread

def plot_cumulative_returns(merged_data, period='5D', title_suffix="", output_path=None):
    """
    Plots cumulative returns for each quantile.
    """
    # Calculate mean return per quantile per day
    daily_quantile_ret = merged_data.groupby(['date', 'factor_quantile'])[period].mean().unstack()
    
    # The 'period' return is the return over N days.
    # To plot cumulative performance properly for overlapping periods (like 5D return reported daily),
    # we usually just sum them or compound them as if they were 1-period returns if we want a "signal performance" metric.
    # Alphalens typically computes the "Cumulative Return of the Factor Weighted Portfolio".
    # For a simple "Quantile Cumulative Return", we can compound the mean daily returns of the quantile.
    # However, since we have N-day returns, compounding them daily assumes we rebalance daily.
    # If we rebalance daily, the return for that day is roughly 1/N * (Return of portfolio formed N days ago + ...).
    # Alphalens simplifies this: it usually looks at 1D returns for cumulative plots.
    # If we want to show "Cumulative Performance" using 5D returns, it's tricky because of overlap.
    # We will use the 1D return column for the Cumulative Plot if available, because that represents the daily realization of the portfolio.
    # If the user specifically asks for "Cumulative performance for each lookahead period", 
    # usually that means: "Show me the alpha curve if I traded based on this horizon."
    # We will stick to compounding the mean return of the quantile for that period? 
    # Let's approximate: 1 + Mean_Return
    
    # Actually, let's treat the 'period' return as the return earned over that period.
    # If we sum them, we get total return.
    
    cum_ret = (1 + daily_quantile_ret).cumprod()
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(cum_ret.columns)))
    
    for i, col in enumerate(cum_ret.columns):
        ax.plot(cum_ret.index, cum_ret[col], label=f'Quintile {col}', color=colors[i], linewidth=2)
        
    ax.set_title(f"Cumulative Performance by Quintile ({period} Horizon) {title_suffix}", fontsize=14, color='white')
    ax.set_ylabel("Cumulative Return (Factor 1.0 start)", color='white')
    ax.set_xlabel("Date", color='white')
    ax.legend(facecolor='black', edgecolor='white')
    ax.grid(True, color='#333333', linestyle='--')
    
    if output_path:
        plt.savefig(output_path, facecolor='black')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
