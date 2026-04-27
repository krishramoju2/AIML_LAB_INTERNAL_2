import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

for c in ['Glucose', 'BloodPressure', 'BMI']:
    X[c] = X[c].replace(0, X[c].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Model,Accuracy")

for d in [3, 5, None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"Decision Tree (depth={d})  {acc}")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"Random Forest            {accuracy_score(y_test, rf.predict(X_test))}")





bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50)
bag.fit(X_train, y_train)
print("Bagging", round(accuracy_score(y_test, bag.predict(X_test)), 4))

ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
print("AdaBoost", round(accuracy_score(y_test, ada.predict(X_test)), 4))

base = [('dt', DecisionTreeClassifier()), ('rf', RandomForestClassifier(50))]
stack = StackingClassifier(base, LogisticRegression())
stack.fit(X_train, y_train)
print("Stacking", round(accuracy_score(y_test, stack.predict(X_test)), 4))
