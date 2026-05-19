import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

#rate of return
data['Return'] = data['Close'].pct_change()

#50 and 200 moving average
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
    
#plotting
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label='Price')
plt.plot(data['MA50'], label='MA50')
plt.plot(data['MA200'], label='MA200')

plt.legend()
plt.title("Moving Average Strategy")
plt.show()

#standard deviation for 20-period rolling
data['Volatility'] = data['Return'].rolling(window=20).std()

data['Volatility'].plot(title="Volatility (Risk)")
plt.show()

#Analyst when buy and sell
data['Signal'] = 0

data.loc[data['MA50'] > data['MA200'], 'Signal'] = 1  # buy=1, sell=0

print(data[['MA50','MA200','Signal']].tail())

#New lesson
#return that if we trade
data['Market_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Signal'].shift(1) * data['Market_Return']

#calculate cumulative return
data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

#plotting
plt.figure(figsize=(10,5))

plt.plot(data['Cumulative_Market'], label='Buy and Hold')
plt.plot(data['Cumulative_Strategy'], label='Strategy')

plt.legend()
plt.title("Strategy vs Market")
plt.show()
