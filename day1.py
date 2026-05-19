import yfinance as yf
import pandas as pd

# get stock data（Apple）
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

print(data.head())

data['Return'] = data['Close'].pct_change()
print(data[['Close', 'Return']].head())


print(data[['Close', 'Return']].head())


import matplotlib.pyplot as plt

data['Close'].plot(title="Stock Price")
plt.show()
