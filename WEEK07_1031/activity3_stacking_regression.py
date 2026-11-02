import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("insurance.csv")

print("Dataset Shape:", df.shape)
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCharges Summary:")
print(df["charges"].describe())

# Encode Sex and Smoker
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

df = pd.get_dummies(
    df, columns=["region"], drop_first=True, dtype=int
)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge()
ridge.fit(X_train_scaled, y_train)

rf = RandomForestRegressor(
    n_estimators=100, random_state=42
)
rf.fit(X_train_scaled, y_train)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train_scaled, y_train)

base_models = [
    ("ridge", Ridge()),
    ("rf", RandomForestRegressor(
        n_estimators=100, random_state=42
    )),
    ("gb", GradientBoostingRegressor(random_state=42))
]

stack = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)

stack.fit(X_train_scaled, y_train)

models = {
    "Ridge": ridge,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Stacking": stack
}

results = []

for name, model in models.items():

    pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Final Estimator Coefficients
print("\nFinal Estimator Coefficients:")

for name, coef in zip(
    ["Ridge", "Random Forest", "Gradient Boosting"],
    stack.final_estimator_.coef_
):
    print(name, ":", coef)

stack_pred = stack.predict(X_test_scaled)

plt.scatter(y_test, stack_pred, alpha=0.6)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()