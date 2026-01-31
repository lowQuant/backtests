import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from dataset_gen import trendline_breakout_dataset
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_prep_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Normalize columns for the dataset generator
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    return df

def get_top_liquid_symbols(df, n=100):
    print(f"Selecting top {n} liquid symbols...")
    # Find the earliest date in the dataset
    min_date = df['Date'].min()
    
    # Filter for data on the first available day (or first week to be safe)
    # Some stocks might not trade on day 1. Let's take the average liquidity of the first month.
    first_month_end = min_date + timedelta(days=30)
    
    start_df = df[df['Date'] < first_month_end]
    
    # Calculate average daily liquidity (close * volume)
    start_df['liquidity'] = start_df['close'] * start_df['volume']
    avg_liq = start_df.groupby('Symbol')['liquidity'].mean()
    
    top_symbols = avg_liq.sort_values(ascending=False).head(n).index.tolist()
    print(f"Top symbols selected: {top_symbols[:5]}...")
    return top_symbols

def train_model(df, train_symbols, train_end_date):
    print("Preparing training data...")
    X_all = []
    y_all = []
    
    # Filter for training period
    train_df_all = df[df['Date'] < train_end_date]
    
    count = 0
    for symbol in train_symbols:
        symbol_df = train_df_all[train_df_all['Symbol'] == symbol].set_index('Date').sort_index()
        
        # Need enough data for lookback (168 + 72 approx)
        if len(symbol_df) < 250: 
            continue
            
        try:
            trades, data_x, data_y = trendline_breakout_dataset(symbol_df, lookback=72)
            if not data_x.empty:
                X_all.append(data_x)
                y_all.append(data_y)
        except Exception as e:
            # print(f"Error processing {symbol}: {e}")
            continue
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(train_symbols)} training symbols")

    if not X_all:
        print("No training data generated!")
        return None

    X_train = pd.concat(X_all)
    y_train = pd.concat(y_all)
    
    print(f"Training Random Forest on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def run_inference(df, model, start_date):
    print("Running inference on all symbols...")
    
    # Filter for testing period
    test_df_all = df[df['Date'] >= start_date]
    
    results = []
    
    unique_symbols = test_df_all['Symbol'].unique()
    # Limit to 20 random symbols to avoid timeout
    if len(unique_symbols) > 20:
        np.random.seed(42)
        unique_symbols = np.random.choice(unique_symbols, 20, replace=False)
        print(f"Downsampling test set to 20 symbols (from {len(test_df_all['Symbol'].unique())}) for performance...")
    
    print(f"Testing on {len(unique_symbols)} symbols...")
    
    count = 0
    for symbol in unique_symbols:
        symbol_df = test_df_all[test_df_all['Symbol'] == symbol].set_index('Date').sort_index()
        
        if len(symbol_df) < 200:
            continue
            
        try:
            # We need to generate the same features
            trades, data_x, data_y = trendline_breakout_dataset(symbol_df, lookback=72)
            
            if data_x.empty:
                continue
                
            # Predict
            probs = model.predict_proba(data_x)[:, 1]
            
            # Filter trades
            trades['model_prob'] = probs
            selected_trades = trades[trades['model_prob'] > 0.5]
            
            # Calculate metrics
            n_trades = len(selected_trades)
            if n_trades > 0:
                avg_ret = selected_trades['return'].mean()
                total_ret = selected_trades['return'].sum()
                win_rate = len(selected_trades[selected_trades['return'] > 0]) / n_trades
            else:
                avg_ret = 0
                total_ret = 0
                win_rate = 0
                
            results.append({
                'Symbol': symbol,
                'N_Trades': n_trades,
                'Avg_Return': avg_ret,
                'Total_Return': total_ret,
                'Win_Rate': win_rate
            })
            
        except Exception as e:
            continue
        
        count += 1
        if count % 500 == 0:
            print(f"Tested {count} symbols...")
            
    return pd.DataFrame(results)

def main():
    data_path = '../data/us_stock_history_10y.csv'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    df = load_and_prep_data(data_path)
    
    # 4 Year Split
    min_date = df['Date'].min()
    split_date = min_date + pd.DateOffset(years=4)
    print(f"Dataset Start: {min_date}, Split Date: {split_date}")
    
    # 1. Select Top 30 Liquid Symbols (at start) - Reduced from 100 for performance
    top_100 = get_top_liquid_symbols(df, 30)
    
    # 2. Train on first 4 years of Top 30
    model = train_model(df, top_100, split_date)
    
    if model:
        # 3. Run on All Symbols (After split date)
        results = run_inference(df, model, split_date)
        
        # Summary
        print("\n=== Results Summary (Out of Sample) ===")
        print(f"Total Symbols Tested: {len(results)}")
        
        # Filter for active symbols (at least 1 trade found)
        active_results = results[results['N_Trades'] > 0]
        print(f"Symbols with Trades: {len(active_results)}")
        
        mean_strategy_return = active_results['Total_Return'].mean()
        mean_avg_trade = active_results['Avg_Return'].mean()
        mean_win_rate = active_results['Win_Rate'].mean()
        
        print(f"Mean Strategy Return per Symbol: {mean_strategy_return:.2%}")
        print(f"Mean Average Trade Return: {mean_avg_trade:.2%}")
        print(f"Mean Win Rate: {mean_win_rate:.2%}")
        
        print("\n--- Top 10 Performing Symbols ---")
        print(active_results.sort_values('Total_Return', ascending=False).head(10)[['Symbol', 'N_Trades', 'Total_Return', 'Win_Rate']])
        
        print("\n--- Bottom 5 Performing Symbols ---")
        print(active_results.sort_values('Total_Return', ascending=True).head(5)[['Symbol', 'N_Trades', 'Total_Return', 'Win_Rate']])

if __name__ == "__main__":
    main()
