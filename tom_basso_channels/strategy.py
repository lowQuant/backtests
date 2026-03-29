"""
Tom Basso All-Weather Trader Strategy
Uses three channel indicators: Donchian, Keltner, and Bollinger Bands

Entry: Buy next day if ANY indicator signals a buy (close > upper band)
Exit: Close position next day if ANY indicator reverses (close < lower band)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def donchian_signal(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int) -> pd.Series:
    """
    Donchian Channel breakout signal (vectorized).
    Buy when close > highest high of lookback period (shifted by 1 to avoid look-ahead).
    Sell when close < lowest low of lookback period.
    """
    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    
    signal = pd.Series(np.nan, index=close.index)
    signal.loc[close > upper] = 1
    signal.loc[close < lower] = -1
    signal = signal.ffill()
    return signal


def keltner_signal(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int, atr_mult: float = 2.0) -> pd.Series:
    """
    Keltner Channel breakout signal (vectorized).
    Uses EMA of close and ATR for the bands.
    Buy when close > EMA + atr_mult * ATR
    Sell when close < EMA - atr_mult * ATR
    """
    ema = close.ewm(span=lookback, adjust=False).mean()
    
    # Calculate ATR
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(true_range, index=close.index).rolling(lookback).mean()
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    # Shift by 1 to avoid look-ahead bias
    upper = upper.shift(1)
    lower = lower.shift(1)
    
    signal = pd.Series(np.nan, index=close.index)
    signal.loc[close > upper] = 1
    signal.loc[close < lower] = -1
    signal = signal.ffill()
    return signal


def bollinger_signal(close: pd.Series, lookback: int, std_mult: float = 2.0) -> pd.Series:
    """
    Bollinger Bands breakout signal (vectorized).
    Uses SMA of close and standard deviation for the bands.
    Buy when close > SMA + std_mult * StdDev
    Sell when close < SMA - std_mult * StdDev
    """
    sma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Shift by 1 to avoid look-ahead bias
    upper = upper.shift(1)
    lower = lower.shift(1)
    
    signal = pd.Series(np.nan, index=close.index)
    signal.loc[close > upper] = 1
    signal.loc[close < lower] = -1
    signal = signal.ffill()
    return signal


def get_combined_signal(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int, 
                        atr_mult: float = 2.0, std_mult: float = 2.0) -> pd.Series:
    """
    Combined signal from all three indicators (vectorized).
    
    Entry Logic: Buy if ANY indicator signals a buy (1)
    Exit Logic: Exit (go flat or short) if ANY indicator signals a sell (-1)
    
    Returns a signal series: 1 = long, -1 = short, 0 = flat
    """
    donchian = donchian_signal(close, high, low, lookback)
    keltner = keltner_signal(close, high, low, lookback, atr_mult)
    bollinger = bollinger_signal(close, lookback, std_mult)
    
    # Create combined signal
    signal = pd.Series(np.nan, index=close.index)
    
    # Buy if ANY indicator is bullish (=1)
    any_buy = (donchian == 1) | (keltner == 1) | (bollinger == 1)
    
    # Sell if ANY indicator is bearish (=-1)
    any_sell = (donchian == -1) | (keltner == -1) | (bollinger == -1)
    
    # Logic: If any buy and no sell -> long (1)
    #        If any sell -> exit/short (-1)
    #        Otherwise maintain previous position
    signal.loc[any_buy & ~any_sell] = 1
    signal.loc[any_sell] = -1
    
    signal = signal.ffill().fillna(0)
    return signal


def calculate_metrics(returns: pd.Series) -> dict:
    """Calculate strategy performance metrics."""
    r = returns.dropna()
    if len(r) == 0 or r[r < 0].abs().sum() == 0:
        return {'profit_factor': np.nan, 'sharpe_ratio': np.nan, 'total_return': np.nan}
    
    profit_factor = r[r > 0].sum() / r[r < 0].abs().sum() if r[r < 0].abs().sum() > 0 else np.inf
    sharpe_ratio = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    total_return = r.sum()
    
    return {
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return
    }


def run_backtest_single_stock(ohlc: pd.DataFrame, lookback: int) -> dict:
    """Run backtest for a single stock with given lookback."""
    signal = get_combined_signal(ohlc['Close'], ohlc['High'], ohlc['Low'], lookback)
    
    # Calculate log returns and shift for next-day execution
    log_return = np.log(ohlc['Close']).diff().shift(-1)
    strategy_return = signal * log_return
    
    return calculate_metrics(strategy_return)


def load_stock_data(data_path: Path = None) -> pd.DataFrame:
    """Load US stock history data."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "mean_reversion" / "us_stock_history_10y.csv"
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    return df


def run_universe_backtest_vectorized(df: pd.DataFrame, lookback: int, 
                                      atr_mult: float = 2.0, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Run vectorized backtest across the entire stock universe.
    
    Args:
        df: DataFrame with columns Date, Symbol, Open, High, Low, Close, Volume
        lookback: Lookback period for all indicators
        atr_mult: ATR multiplier for Keltner Channels
        std_mult: Std dev multiplier for Bollinger Bands
    
    Returns:
        DataFrame with Date, Symbol, signal, log_return, strategy_return columns
    """
    # Sort by symbol and date for proper groupby operations
    df = df.sort_values(['Symbol', 'Date']).copy()
    
    # Calculate signals per symbol using groupby transform
    def calc_signal_group(g):
        return get_combined_signal(g['Close'], g['High'], g['Low'], lookback, atr_mult, std_mult)
    
    df['signal'] = df.groupby('Symbol', group_keys=False).apply(calc_signal_group, include_groups=False)
    
    # Calculate log returns per symbol (shift within each symbol group)
    df['log_return'] = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x).diff().shift(-1))
    
    # Strategy return = signal * next day's log return
    df['strategy_return'] = df['signal'] * df['log_return']
    
    return df


def calculate_profit_factor_from_mean_returns(df: pd.DataFrame) -> float:
    """
    Calculate profit factor from mean returns across symbols by date.
    
    Groups by Date, calculates mean strategy_return across all symbols,
    then computes profit factor from these mean returns.
    """
    # Calculate mean return per date across all symbols
    mean_returns = df.groupby('Date')['strategy_return'].mean().dropna()
    
    if len(mean_returns) == 0:
        return np.nan
    
    gains = mean_returns[mean_returns > 0].sum()
    losses = mean_returns[mean_returns < 0].abs().sum()
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    return gains / losses


def run_universe_backtest(lookback: int, data_path: Path = None) -> dict:
    """
    Run backtest across the entire stock universe (legacy interface).
    Returns aggregated metrics using mean returns across symbols by date.
    """
    df = load_stock_data(data_path)
    
    # Filter out invalid prices
    df = df[df['Close'] > 0].copy()
    
    # Run vectorized backtest
    result_df = run_universe_backtest_vectorized(df, lookback)
    
    # Calculate mean returns per date
    mean_returns = result_df.groupby('Date')['strategy_return'].mean().dropna()
    
    # Calculate metrics from mean returns
    metrics = calculate_metrics(mean_returns)
    
    return metrics


if __name__ == "__main__":
    # Quick test with a single lookback
    print("Testing Tom Basso Channel Strategy...")
    
    metrics = run_universe_backtest(lookback=20)
    print(f"Lookback: 20")
    print(f"  Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Total Log Return: {metrics['total_return']:.4f}")
