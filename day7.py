import yfinance as yf
import pandas as pd


data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

#cuurent close price - previous close price
delta = data['Close'].diff()

#let negative number to 0 to calculate values of upward
gain = delta.clip(lower=0)
#let positive number to 0 to calculate values of downward
loss = -delta.clip(upper=0)

#calculate average gain and loss
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

#RS and RSI formula
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))


#prepare variable needed
data['Return'] = data['Close'].pct_change()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

#prepare features needed, Add RSI
features = ['MA50', 'MA200', 'Return', 'RSI']
X = data[features]
y = data['Target']

#delete NA
X = X.dropna()
y = y.loc[X.index]

#Split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False 
    )

#build Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
#train model and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Accuracy of model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

    
#Strategy Return
#Building Result DataFrame
results = X_test.copy()
results['Prediction'] = predictions
results['Actual_Return'] = data.loc[X_test.index, 'Return']

#calculate AI prediction's Return
results['AI_Strategy_Return'] = (
    results['Prediction'].shift(1)
    * results['Actual_Return']
)

#calculate Cumulative Return of AI Stratagy
results['AI_Cumulative'] = (
    1 + results['AI_Strategy_Return']
).cumprod()

#calculate Cumulative Return of Buy & Hold
results['Buy_Hold'] = (
    1 + results['Actual_Return']
).cumprod()

#Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(results['AI_Cumulative'], label='AI Strategy')
plt.plot(results['Buy_Hold'], label='Buy and Hold')
plt.legend()
plt.title("AI Strategy vs Buy and Hold")
plt.show()

#Calculate total profit
ai_return = results['AI_Cumulative'].iloc[-1]
market_return = results['Buy_Hold'].iloc[-1]
print("AI Strategy:", ai_return)
print("Buy & Hold:", market_return)

#AI predicts stock tommorow up or down
latest_data = X.iloc[-1:]
prediction = model.predict(latest_data)
print("BUY OR NOT:",prediction)

#AI predicts probability stock tommorow up or down
probability = model.predict_proba(latest_data)
print("DOWN OR UP:",probability)