# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import numpy as np
import pandas as pd

data = {
    "Name": ["Alice", "Bob", None, "David", "Eva"],
    "Age": [25, 30, None, 35, 28],
    "City": ["New York", "London", "Tokyo", None, "Paris"],
    "Salary": [50000, 60000, 70000, 80000, None]
}

df = pd.DataFrame(data)
print("DataFrame with Missing Values:")
print(df)

df_dropna_rows = df.dropna(axis=0)  # Drop rows with any missing values
print("\nDataFrame after dropping rows with missing values:")
print(df_dropna_rows)


df_dropna_cols = df.dropna(axis=1)  # Drop columns with any missing values
print("\nDataFrame after dropping columns with missing values:")
print(df_dropna_cols)