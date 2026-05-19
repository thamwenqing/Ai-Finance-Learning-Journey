import yfinance as yf

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

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

data['Volatility'] = (
    data['Return']
    .rolling(window=10)
    .std()
)

data['Momentum_5D'] = (
    data['Close'] / data['Close'].shift(5) - 1
)

data['Return_Lag_1'] = (
    data['Return'].shift(1)
)

data['Return_Lag_2'] = (
    data['Return'].shift(2)
)

features = [
    'MA50',
    'MA200',
    'RSI',
    'Volatility',
    'Momentum_5D',
    'Return_Lag_1',
    'Return_Lag_2'
]

X = data[features]
y = data['Target']

X = X.dropna()
y = y.loc[X.index]

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_index, test_index in tscv.split(X):
   
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)

    scores.append(score)

    print("Fold Accuracy:", score)