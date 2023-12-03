from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTreeClassifier

df = datasets.load_breast_cancer()
X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12345)

model = DecisionTreeClassifier(max_depth=32)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, y_pred)
print(acc)