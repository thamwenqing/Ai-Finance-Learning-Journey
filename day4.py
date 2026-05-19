import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

data['Return'] = data['Close'].pct_change()

data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

#prepare features needed
features = ['MA50', 'MA200', 'Return']
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

#Build AI model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Accurary for the prediction
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

