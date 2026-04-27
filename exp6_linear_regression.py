import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

for col in ['Glucose', 'BloodPressure', 'BMI']:
    X[col] = X[col].replace(0, X[col].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

models = [
    ('Least Squares', LinearRegression(), False),
    ('Gradient Descent', SGDRegressor(), False),
    ('Ridge', Ridge(), False),
    ('LASSO', Lasso(), False),
    ('Polynomial', LinearRegression(), True)
]

print(f"{'Model':<20} {'MSE':<12} {'R2 Score':<10}")
for name, model, use_poly in models:
    if use_poly:
        model.fit(X_train_poly, y_train)
        preds = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    mse_val = mean_squared_error(y_test, preds)
    r2_val = r2_score(y_test, preds)
    print(f"{name:<20} {mse_val:<12.4f} {r2_val:<10.4f}")
