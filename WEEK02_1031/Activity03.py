import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
degrees_to_test = [2, 3, 4]

df = pd.read_csv("Car Price Prediction.csv")

print("--- Dataset Info ---")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())

df = df.dropna()
X = df[["enginesize", "horsepower", "wheelbase", "carlength"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_rmse = np.sqrt(mean_squared_error(y_test, linear_prediction))
linear_r2 = r2_score(y_test, linear_prediction)

print("\nLinear Regression")
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)
results = []

for degree in degrees_to_test:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    prediction = model.predict(X_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    results.append([degree, rmse, r2])

    print(f"\nPolynomial Regression Degree {degree}")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

results_df = pd.DataFrame(
    results,
    columns=["Degree", "RMSE", "R2 Score"]
)

print("\nComparison:")
print(results_df)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, linear_prediction, label="Linear", alpha=0.6)

for degree in degrees_to_test:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    prediction = model.predict(X_test_poly)
    plt.scatter(y_test, prediction, label=f"Polynomial {degree}", alpha=0.6)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))

x = df["enginesize"].values
y_price = df["price"].values
x_sorted = np.sort(x)

for degree in degrees_to_test:
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, y_price)

    x_curve = poly.transform(x_sorted.reshape(-1, 1))
    y_curve = model.predict(x_curve)

    plt.plot(x_sorted, y_curve, label=f"Degree {degree}")

plt.scatter(x, y_price, alpha=0.4, label="Actual Data", color='gray')
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Polynomial Regression Curves (Engine Size vs Price)")
plt.legend()
plt.grid(True)
plt.show()
