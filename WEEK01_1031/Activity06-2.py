# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# 1. Load the Breast Cancer dataset
# ---------------------------------------------------------

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Dataset Shape:")
print(df.shape)


# ---------------------------------------------------------
# 2. Detect outliers using IQR
#    Features: mean radius and mean texture
# ---------------------------------------------------------

features = ["mean radius", "mean texture"]

for feature in features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[feature] < lower_bound) |
        (df[feature] > upper_bound)
    ]

    print("\nOutlier Analysis for:", feature)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Number of Outliers:", len(outliers))

    # Boxplot
    plt.figure(figsize=(7, 5))
    plt.boxplot(df[feature])
    plt.ylabel(feature)
    plt.title("Boxplot of " + feature)
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# 3. Log transformation of mean area
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["mean area"], bins=20, edgecolor="black")
plt.xlabel("Mean Area")
plt.ylabel("Frequency")
plt.title("Mean Area Before Log Transformation")
plt.grid(True)
plt.show()


# Apply log transformation
df["log mean area"] = np.log1p(df["mean area"])


plt.figure(figsize=(8, 5))
plt.hist(df["log mean area"], bins=20, edgecolor="black")
plt.xlabel("Log Mean Area")
plt.ylabel("Frequency")
plt.title("Mean Area After Log Transformation")
plt.grid(True)
plt.show()


print("\nLog Transformation:")
print("The log transformation reduces the effect of very large values")
print("and makes the distribution of mean area less right-skewed.")


# ---------------------------------------------------------
# 4. Feature Scaling using StandardScaler
# ---------------------------------------------------------

X = df[data.feature_names]
y = df["target"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)

print("\nScaling Explanation:")
print("StandardScaler puts features on a similar scale.")
print("This is important for distance-based models such as KNN")
print("and SVM because large-scale features can otherwise dominate.")


# ---------------------------------------------------------
# 5. PCA - Reduce to 2 components
# ---------------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:")
print(X_pca.shape)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(pca.explained_variance_ratio_.sum())


# PCA scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Malignant"
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Benign"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Breast Cancer Dataset")
plt.legend()
plt.grid(True)
plt.show()


print("\nPCA Comment:")
print("The two classes show noticeable separation in the PCA plot,")
print("although some points from the two classes overlap.")


# ---------------------------------------------------------
# 6. Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 7. SVM Classifier
# ---------------------------------------------------------

svm_model = SVC(kernel="linear")

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)


# ---------------------------------------------------------
# 8. KNN Classifier
# ---------------------------------------------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)


# ---------------------------------------------------------
# 9. Compare Accuracy
# ---------------------------------------------------------

print("\n========== CLASSIFIER RESULTS ==========")

print("SVM Accuracy:", round(svm_accuracy, 4))
print("KNN Accuracy:", round(knn_accuracy, 4))

if svm_accuracy > knn_accuracy:
    print("SVM has higher accuracy on this test split.")
elif knn_accuracy > svm_accuracy:
    print("KNN has higher accuracy on this test split.")
else:
    print("Both classifiers have the same accuracy.")


# ---------------------------------------------------------
# 10. Final Preprocessing Comment
# ---------------------------------------------------------

print("\n========== FINAL COMMENT ==========")

print(
    "Scaling had the biggest impact because the dataset contains "
    "features with very different numerical ranges."
)

print(
    "Standardization prevents large-scale features from dominating "
    "distance calculations, which is especially important for KNN and SVM."
)