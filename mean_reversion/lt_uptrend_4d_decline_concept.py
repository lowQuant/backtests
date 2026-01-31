from symtable import SymbolTable
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import arcticdb as adb
import os
# import dotenv

# dotenv.load_dotenv()

# bucket_name = os.getenv("BUCKET_NAME")
# aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_region = os.getenv("AWS_REGION")

# ac = adb.Arctic(f's3://s3.{aws_region}.amazonaws.com:{bucket_name}?region={aws_region}&access={aws_access_key_id}&secret={aws_secret_access_key}')


# print(ac.get_library("us_equities").read("AAPL").data)

data = yf.download(["AAPL","META","MSFT"],period="10y",group_by="ticker",multi_level_index=True)

df = data.stack(level=0).rename_axis(['Date', 'Symbol']).reset_index(level=1)
df = df.sort_values(by='Symbol',axis='index',kind='stable')

# Process each symbol separately
results = []
for symbol in df['Symbol'].unique():
    symbol_df = df[df['Symbol'] == symbol].copy()
    
    # Calculate log returns
    symbol_df['log_returns'] = np.log(symbol_df['Close']) - np.log(symbol_df['Close'].shift(1))
    
    # Calculate next day's return (forward looking)
    symbol_df['next_day_return'] = symbol_df['log_returns'].shift(-1) # np.log(symbol_df['Close'].shift(-1)) - np.log(symbol_df['Open'].shift(-1)) #
    
    # Calculate 200-day moving average
    symbol_df['ma_200'] = symbol_df['Close'].rolling(window=200).mean()
    
    # Count consecutive negative returns
    symbol_df['is_negative'] = (symbol_df['log_returns'] < 0).astype(int)
    
    # Calculate consecutive decline streak
    symbol_df['consecutive_decline'] = 0
    streak = 0
    consecutive_declines = []

    
    
    for idx, is_neg in enumerate(symbol_df['is_negative'].values):
        if is_neg == 1:
            streak += 1
        else:
            streak = 0
        consecutive_declines.append(streak)
    
    symbol_df['consecutive_decline'] = consecutive_declines
    
    # Create signal: 1 if Close > 200d MA AND consecutive decline >= 4
    # Only signal on the 4th day (not 5th, 6th, etc.) to avoid overlapping trades
    symbol_df['signal'] = ((symbol_df['Close'] > symbol_df['ma_200']) & 
                           (symbol_df['consecutive_decline'] == 4)).astype(int)
    
    # Strategy return: next day's return when signal == 1, else 0
    symbol_df['strat_return'] = symbol_df['signal'] * symbol_df['next_day_return']
    print(symbol_df[symbol_df['signal'] == 1]['strat_return'].describe())
    
    results.append(symbol_df)

# Combine all symbols back together
df = pd.concat(results, axis=0)

# Display results where signal == 1
signal_occurrences = df[df['signal'] == 1][['Symbol', 'Close', 'ma_200', 'log_returns', 'consecutive_decline', 'next_day_return', 'signal']]

print("\n" + "="*80)
print("OCCURRENCES: Exactly 4 Consecutive Down Days While Above 200-Day MA")
print("="*80)
print(f"\nTotal occurrences found: {len(signal_occurrences)}")

# Calculate average next day return by symbol
print("\n" + "="*80)
print("AVERAGE NEXT DAY RETURN BY SYMBOL")
print("="*80)
for symbol in df['Symbol'].unique():
    symbol_signals = signal_occurrences[signal_occurrences['Symbol'] == symbol]
    if len(symbol_signals) > 0:
        avg_return = symbol_signals['next_day_return'].mean()
        count = len(symbol_signals)
        win_rate = (symbol_signals['next_day_return'] > 0).sum() / count * 100
        print(f"{symbol}: {count} trades, Avg Return: {avg_return:.4f} ({avg_return*100:.2f}%), Win Rate: {win_rate:.1f}%")
    else:
        print(f"{symbol}: No trades")

# Calculate strategy statistics
print("\n" + "="*80)
print("STRATEGY STATISTICS (ALL SYMBOLS COMBINED)")
print("="*80)

# Get trade-only returns (days when we actually trade)
all_strat_returns = df[df['strat_return'] != 0]['strat_return'].dropna()

if len(all_strat_returns) > 0:
    # Number of trades and unique signal days
    num_trades = len(all_strat_returns)
    unique_signal_days = df[df['signal'] == 1].index.nunique()
    
    # Trade statistics (using trade-only log returns)
    avg_return_per_trade = all_strat_returns.mean()
    win_rate = (all_strat_returns > 0).sum() / len(all_strat_returns) * 100
    
    # Profit factor: sum of gains / sum of losses
    gains = all_strat_returns[all_strat_returns > 0].sum()
    losses = abs(all_strat_returns[all_strat_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.nan
    
    # Cumulative return: exp(sum of log returns) - 1
    cumulative_return = np.exp(all_strat_returns.sum()) - 1
    
    # Annualized return: exp(mean log return * 252) - 1
    annualized_return = np.exp(avg_return_per_trade * 252) - 1
    
    # Annualized volatility: std(log returns) * sqrt(252)
    annualized_volatility = all_strat_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (annualized): mean * sqrt(252) / std
    sharpe_ratio = (avg_return_per_trade * np.sqrt(252)) / all_strat_returns.std() if all_strat_returns.std() > 0 else np.nan
    
    # Daily Sharpe ratio (non-annualized, per trade): mean / std
    daily_sharpe = avg_return_per_trade / all_strat_returns.std() if all_strat_returns.std() > 0 else np.nan
    
    # Max drawdown using trade-only equity curve
    equity_curve = np.exp(all_strat_returns.cumsum())  # starts at 1
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"Number of Trades: {num_trades}")
    print(f"Unique Days with Signals: {unique_signal_days}")
    print(f"Average Return per Trade: {avg_return_per_trade:.4f} ({avg_return_per_trade*100:.2f}%)")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profit Factor: {profit_factor:.3f}")
    print(f"Cumulative Return (Compounded): {cumulative_return:.4f} ({cumulative_return*100:.2f}%)")
    print(f"Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
    print(f"Annualized Volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.3f}")
    print(f"Sharpe Ratio (Per Trade): {daily_sharpe:.3f}")
    print(f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
else:
    print("No trades found")

print("\n" + "="*80)
