# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd
#create dataframe for dictionary

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "London", "Tokyo"],
    "Salary": [50000, 60000, 70000]
}

df = pd.DataFrame(data)
print("Manual DataFrame:")
print(df)