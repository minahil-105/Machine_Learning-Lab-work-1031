import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from xgboost import XGBClassifier

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Drop Customer ID
df.drop("customerID", axis=1, inplace=True)

# Encode Target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

numeric_cols = X_train.select_dtypes(include=["number"]).columns
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

gb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

# AdaBoost
ada = Pipeline([
    ("preprocessor", preprocessor),
    ("model", AdaBoostClassifier(random_state=42))
])

xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

# Train Models
gb.fit(X_train, y_train)
ada.fit(X_train, y_train)
xgb.fit(X_train, y_train)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
    "model__subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest F1 Score:")
print(grid.best_score_)

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }

results = pd.DataFrame({
    "Gradient Boosting": evaluate(gb, X_test, y_test),
    "AdaBoost": evaluate(ada, X_test, y_test),
    "Default XGBoost": evaluate(xgb, X_test, y_test),
    "Tuned XGBoost": evaluate(best_xgb, X_test, y_test)
}).T

print("\nModel Comparison:")
print(results)

# Feature Importance
feature_names = best_xgb.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = best_xgb.named_steps[
    "model"
].feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

top5 = feature_df.sort_values(
    by="Importance", ascending=False
).head(5)

print("\nTop 5 Important Features:")
print(top5)

plt.barh(top5["Feature"], top5["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 5 Feature Importance")
plt.gca().invert_yaxis()
plt.show()