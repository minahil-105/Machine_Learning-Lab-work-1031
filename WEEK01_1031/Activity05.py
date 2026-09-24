# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd
import matplotlib.pyplot as plt

# Activity 5: Data Cleaning & Outlier Detection

# 1. Create the dataset
data = {
    "Name": ["Shahramn", "Hina", "Ayeza", "Zara", "Mariha", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 2. Handle missing values

# Fill missing Age with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Salary with the median salary
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

print("\nDataset after handling missing values:")
print(df)

# Check if any missing values remain
print("\nMissing values:")
print(df.isnull().sum())

# 3. Encode Department using One-Hot Encoding

df_encoded = pd.get_dummies(
    df,
    columns=["Department"],
    dtype=int
)

print("\nDataset after One-Hot Encoding:")
print(df_encoded)

# 4. Detect salary outliers using a boxplot

plt.figure(figsize=(8, 5))

plt.boxplot(df["Salary"])

plt.ylabel("Salary")
plt.title("Salary Outlier Detection")
plt.grid(True)

plt.show()

# 5. Detect outliers using IQR

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
]

print("\nSalary Outlier Analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

print("\nDetected Salary Outliers:")
print(outliers[["Name", "Salary"]])

# 6. Comment on Zara's salary

zara_salary = df.loc[df["Name"] == "Zara", "Salary"].iloc[0]

if zara_salary > upper_bound:
    print("\nComment:")
    print("Zara's salary is statistically an outlier according to the IQR method.")
else:
    print("\nComment:")
    print("Zara's salary is not statistically identified as an outlier.")