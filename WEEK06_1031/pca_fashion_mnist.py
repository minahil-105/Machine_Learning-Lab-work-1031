
# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Fashion-MNIST dataset
print("Loading Fashion-MNIST dataset...")

X, y = fetch_openml(
    "Fashion-MNIST",
    version=1,
    return_X_y=True,
    as_frame=False,
    parser="auto"
)

X = X.astype("float32")
y = y.astype(int)

# Use a subset to make the experiment faster
X = X[:10000]
y = y[:10000]

print("Dataset loaded successfully!")
print("Original number of features:", X.shape[1])
print("Number of images:", X.shape[0])

# Step 3: Scale pixel values
X = X / 255.0

# Step 4: Apply PCA using three approaches
pca_50 = PCA(n_components=50)
X_50 = pca_50.fit_transform(X)

pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X)

pca_all = PCA(n_components=None)
X_all = pca_all.fit_transform(X)

# Step 5: Display results
results = [
    {
        "Method": "Fixed 50 Components",
        "Original Features": X.shape[1],
        "Retained Components": X_50.shape[1],
        "Cumulative Explained Variance (%)":
            np.sum(pca_50.explained_variance_ratio_) * 100
    },
    {
        "Method": "95% Variance",
        "Original Features": X.shape[1],
        "Retained Components": X_95.shape[1],
        "Cumulative Explained Variance (%)":
            np.sum(pca_95.explained_variance_ratio_) * 100
    },
    {
        "Method": "All Components",
        "Original Features": X.shape[1],
        "Retained Components": X_all.shape[1],
        "Cumulative Explained Variance (%)":
            np.sum(pca_all.explained_variance_ratio_) * 100
    }
]

results_df = pd.DataFrame(results)

print("\nPCA Comparison Table:")
print(results_df.to_string(index=False, formatters={
    "Cumulative Explained Variance (%)": "{:.2f}".format
}))

# Step 6: Plot cumulative explained variance
plt.figure(figsize=(9, 6))

plt.plot(
    np.arange(1, len(pca_50.explained_variance_ratio_) + 1),
    np.cumsum(pca_50.explained_variance_ratio_),
    label="50 Components"
)

plt.plot(
    np.arange(1, len(pca_95.explained_variance_ratio_) + 1),
    np.cumsum(pca_95.explained_variance_ratio_),
    label="95% Variance"
)

plt.plot(
    np.arange(1, len(pca_all.explained_variance_ratio_) + 1),
    np.cumsum(pca_all.explained_variance_ratio_),
    label="All Components"
)

plt.axhline(y=0.95, linestyle="--", label="95% Variance Threshold")

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Visualize the first two principal components
# Fit a separate PCA with two components for a clear 2D plot
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y,
    cmap="tab10",
    s=10,
    alpha=0.6
)

colorbar = plt.colorbar(scatter, ticks=range(10))
colorbar.ax.set_yticklabels(class_names)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Fashion-MNIST Visualized Using Two PCA Components")
plt.tight_layout()
plt.show()

print("\nVariance explained by first two components:",
      round(np.sum(pca_2d.explained_variance_ratio_) * 100, 2), "%")

# Step 8: Final interpretation
print("\nInterpretation:")
print("Method 1: Keeps exactly 50 components and reduces dimensionality.")
print("Method 2: Keeps enough components to explain at least 95% variance.")
print("Method 3: Keeps all components and preserves all variance available")
print("         in the input data, apart from numerical precision.")
