# 23-NTU-CS-1031
# HAFIZA mINAHIL sHABBIR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Activity 1: Pakistani Provinces Analysis

# 1. Create dataset
data = {
    "Province": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "Population (millions)": [127.7, 55.7, 40.9, 14.9],
    "Literacy Rate (%)": [64.7, 57.5, 55.1, 55.5],
    "Region": ["East", "South", "North", "West"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 2. Add missing values for demonstration
df.loc[1, "Literacy Rate (%)"] = None
df.loc[3, "Population (millions)"] = None

print("\nDataset with missing values:")
print(df)

# 3. Handle missing values
# Drop one row containing a missing literacy rate
df = df.dropna(subset=["Literacy Rate (%)"])

# Fill the remaining missing population value with the median
median_population = df["Population (millions)"].median()
df["Population (millions)"] = df["Population (millions)"].fillna(median_population)

print("\nDataset after handling missing values:")
print(df)

# 4. Label Encoding
label_encoder = LabelEncoder()
df["Region_Label"] = label_encoder.fit_transform(df["Region"])

print("\nAfter Label Encoding:")
print(df)

print("\nRegion Mapping:")
for number, region in enumerate(label_encoder.classes_):
    print(number, "=", region)

# 5. One-Hot Encoding
df = pd.get_dummies(df, columns=["Region"], dtype=int)

print("\nAfter One-Hot Encoding:")
print(df)

# 6. Scatter plot: Population vs Literacy Rate
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Population (millions)"],
    df["Literacy Rate (%)"]
)

for i in range(len(df)):
    plt.annotate(
        df["Province"].iloc[i],
        (
            df["Population (millions)"].iloc[i],
            df["Literacy Rate (%)"].iloc[i]
        )
    )

plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate of Pakistani Provinces")
plt.grid(True)
plt.show()

# 7. Detect literacy-rate outliers using IQR
Q1 = df["Literacy Rate (%)"].quantile(0.25)
Q3 = df["Literacy Rate (%)"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Literacy Rate (%)"] < lower_bound) |
    (df["Literacy Rate (%)"] > upper_bound)
]

print("\nOutlier Detection:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

if outliers.empty:
    print("No province is identified as a literacy-rate outlier.")
else:
    print("Literacy-rate outliers:")
    print(outliers[["Province", "Literacy Rate (%)"]])
