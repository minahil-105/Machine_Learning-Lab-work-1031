
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and explore dataset
data = pd.read_csv("train.csv")

print("Shape:", data.shape)
print(data.head())
print("\nMissing values:\n", data.isnull().sum().head())
print("\nSalePrice summary:\n", data["SalePrice"].describe())

# 2. Prepare features and target
X = data.drop(columns=["SalePrice", "Id"])
y = data["SalePrice"]

numeric = X.select_dtypes(include=np.number).columns
categorical = X.select_dtypes(exclude=np.number).columns

# 3. Preprocess missing values and categorical data
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train default Random Forest
model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)
pred_default = model.predict(X_test)

# 6. Tune hyperparameters with GridSearchCV
params = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    model, params, cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred_best = best_model.predict(X_test)

print("\nBest parameters:", grid.best_params_)
print("Best CV MSE:", round(-grid.best_score_, 2))
print("Best CV RMSE:", round(np.sqrt(-grid.best_score_), 2))

# 7. Compare model performance
def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return [
        mean_absolute_error(actual, predicted),
        mse,
        np.sqrt(mse),
        r2_score(actual, predicted)
    ]

results = pd.DataFrame(
    [evaluate(y_test, pred_default), evaluate(y_test, pred_best)],
    columns=["MAE", "MSE", "RMSE", "R2"],
    index=["Default Random Forest", "Tuned Random Forest"]
)

print("\nModel Comparison:\n", results.round(3))

# 8. Plot actual versus predicted prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred_best, alpha=0.6)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 9. Display top five feature importances
prep = best_model.named_steps["prep"]
rf = best_model.named_steps["rf"]
names = prep.get_feature_names_out()

importance = pd.Series(
    rf.feature_importances_, index=names
).sort_values(ascending=False).head(5)

print("\nTop 5 Important Features:\n", importance)

importance.sort_values().plot(kind="barh", figsize=(7, 4))
plt.title("Top 5 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()