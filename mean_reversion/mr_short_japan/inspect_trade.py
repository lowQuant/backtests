import pandas as pd
import b_short_spike_jp as strategy

def inspect_specific_trade():
    symbol = "2998.T"
    entry_date_str = "2023-06-02"
    
    print(f"Inspecting Trade: {symbol} Entry around {entry_date_str}")
    
    # 1. Load Data
    symbols = [symbol]
    df = strategy.download_and_cache_data(symbols)
    if df.empty:
        print("Data not found.")
        return

    # 2. Calc Indicators
    df = strategy.prepare_data(df)
    
    # Filter for the specific symbol
    data = df.xs(symbol, level='Symbol').copy()
    
    # Define range: Entry Date -> +15 days
    start_date = pd.Timestamp(entry_date_str)
    end_date = start_date + pd.Timedelta(days=15)
    
    subset = data[(data.index >= start_date) & (data.index <= end_date)][['Open', 'High', 'Low', 'Close', 'EMA_Exit']]
    
    print("\nDaily Data Trace:")
    print(f"{ 'Date':<12} | { 'Open':<8} | { 'High':<8} | { 'Low':<8} | { 'Close':<8} | { 'EMA(5)':<8} | { 'Close < EMA?':<12} | {'Action'}")
    print("-" * 100)
    
    position_open = True
    
    for date, row in subset.iterrows():
        d_str = date.strftime('%Y-%m-%d')
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        ema = row['EMA_Exit']
        
        condition = c < ema
        
        action = ""
        if position_open:
            if condition:
                action = "EXIT SIGNAL (Target Met) -> Execute at CLOSE"
                position_open = False
            else:
                action = "HOLD"
        else:
            action = "-"
            
        print(f"{d_str:<12} | {o:<8.1f} | {h:<8.1f} | {l:<8.1f} | {c:<8.1f} | {ema:<8.1f} | {str(condition):<12} | {action}")

if __name__ == "__main__":
    inspect_specific_trade()
