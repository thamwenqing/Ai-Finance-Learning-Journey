import yfinance as yf

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
#add max_depth in model(restict depth of tree model)
train_scores = []
test_scores = []
depths = [1,2,3,5,10] #create diff depths

#for single depths
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(
#   n_estimators=100,
#   max_depths=3,#new add in this topic
#   random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
for depth in depths:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )
#train model and predict
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
##Accuracy of train
    train_acc = accuracy_score(y_train, train_pred)
##print("Train Accuracy:", train_acc)
#Accuracy of test
    test_acc = accuracy_score(y_test, test_pred)
##print("Train Accuracy:", test_acc)
    train_scores.append(train_acc)
    test_scores.append(test_acc)

#plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(depths, train_scores, label='Train Accuracy')
plt.plot(depths, test_scores, label='Test Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Overfitting Analysis")
plt.show()