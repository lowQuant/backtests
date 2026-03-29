import yfinance as yf
import pandas as pd

print(f"YFinance Version: {yf.__version__}")
try:
    data = yf.download("SPY", period="5d", progress=False)
    print("Download result:")
    print(data.head())
    if data.empty:
        print("Data is empty!")
    else:
        print("Data OK")
except Exception as e:
    print(f"Error: {e}")
