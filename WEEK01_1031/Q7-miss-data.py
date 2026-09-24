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