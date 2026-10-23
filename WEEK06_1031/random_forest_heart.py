
# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 2: Load the Heart Disease dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

data = pd.read_csv(
    url,
    header=None,
    names=column_names,
    na_values="?"
)

print("Dataset loaded successfully!")
print("\nDataset Shape:")
print(data.shape)

print("\nFirst 10 Rows:")
print(data.head(10))

print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Clean the target and convert it to binary
# 0 = no heart disease, 1-4 = heart disease
data["target"] = pd.to_numeric(data["target"], errors="coerce")
data = data.dropna(subset=["target"])
data["target"] = (data["target"] > 0).astype(int)

# Step 4: Separate features and target
X = data.drop(columns=["target"])
y = data["target"]

print("\nTarget Class Distribution:")
print(y.value_counts())

print("\nTarget Class Distribution (Percentages):")
print((y.value_counts(normalize=True) * 100).round(2))

# Step 5: Ensure all input features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 7: Train the basic Random Forest classifier
basic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(random_state=42))
])

basic_model.fit(X_train, y_train)

basic_predictions = basic_model.predict(X_test)
basic_accuracy = accuracy_score(y_test, basic_predictions)

print("\nBasic Random Forest Test Accuracy:")
print(f"{basic_accuracy:.4f} ({basic_accuracy * 100:.2f}%)")

# Step 8: Define the GridSearchCV parameter grid
parameter_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2]
}

# Step 9: Apply GridSearchCV with 5-fold cross-validation
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV...")
grid_search.fit(X_train, y_train)

# Step 10: Display best hyperparameters and CV accuracy
print("\nBest Hyperparameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(f"{grid_search.best_score_:.4f}")

# Step 11: Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nBest Model Test Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Step 12: Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"]
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix - Best Random Forest")
plt.tight_layout()
plt.show()

# Step 13: Calculate feature importances
forest = best_model.named_steps["rf"]

importance = pd.Series(
    forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Three Most Influential Features:")
print(importance.head(3))

# Step 14: Plot the top 10 feature importances
plt.figure(figsize=(9, 6))

importance.head(10).sort_values().plot(kind="barh")

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# Step 15: Compare the baseline and tuned models
print("\nModel Comparison:")
print(f"Basic Random Forest Accuracy: {basic_accuracy:.4f}")
print(f"Tuned Random Forest Accuracy: {accuracy:.4f}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

print("\nProgram completed successfully!")
