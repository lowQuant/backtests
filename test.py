import yfinance as yf

aapl = yf.download("AAPL", period="1y")
print(aapl.head())