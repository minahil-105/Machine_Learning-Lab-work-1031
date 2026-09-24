import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from collections import Counter

# Load the dataset
data = load_breast_cancer()

# Explore the dataset
print("Data Keys:")
print(data.keys())

print("\nTarget Names:")
print(data.target_names)

print("\nNumber of Samples:")
print(data.data.shape[0])

print("\nNumber of Features:")
print(data.data.shape[1])

print("\nFeature Names:")
print(data.feature_names)

print("\nShape of data.data:")
print(data.data.shape)

print("\nShape of data.target:")
print(data.target.shape)


# Convert dataset into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

print("\nFirst 5 rows:")
print(df.head())


# Check missing values before adding any
print("\nMissing Values Before:")
print(df.isnull().sum().sum())


# Artificially introduce 5 missing values
np.random.seed(42)

random_rows = np.random.choice(df.index, 5, replace=False)
df.loc[random_rows, "mean radius"] = np.nan

print("\nMissing Values After Artificially Adding Them:")
print(df["mean radius"].isnull().sum())


# Handle missing values using median
median_value = df["mean radius"].median()

df["mean radius"] = df["mean radius"].fillna(median_value)

print("\nMissing Values After Handling:")
print(df["mean radius"].isnull().sum())

print("\nMedian used for Mean Radius:")
print(median_value)

print("\nReason:")
print("Median is used because it is less affected by extreme values and outliers.")


# Check class distribution
print("\nClass Distribution:")
class_counts = df["target"].value_counts().sort_index()

print("0 =", data.target_names[0], ":", class_counts[0])
print("1 =", data.target_names[1], ":", class_counts[1])


# Check whether dataset is imbalanced
majority = class_counts.max()
minority = class_counts.min()

imbalance_ratio = majority / minority

print("\nImbalance Ratio:", round(imbalance_ratio, 2))

if imbalance_ratio > 1.5:
    print("The dataset is imbalanced.")
else:
    print("The dataset is not strongly imbalanced.")


# Apply class weighting if dataset is imbalanced
if imbalance_ratio > 1.5:

    class_weights = {}

    total = len(df)
    number_of_classes = len(class_counts)

    for class_value, count in class_counts.items():
        class_weights[class_value] = total / (number_of_classes * count)

    print("\nClass Weights:")
    print(class_weights)

    print("\nClass Counts Before Class Weighting:")
    print(class_counts)

    print("\nClass weighting is applied instead of SMOTE.")
    print("Class weighting gives more importance to the minority class during model training.")

    # Example:
    # model = LogisticRegression(class_weight=class_weights)