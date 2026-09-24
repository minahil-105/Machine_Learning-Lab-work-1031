# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import numpy as np
import pandas as pd

#creating sample with numpy 

np.random.seed(0)  # For reproducibility
# Create a sample DataFrame with random data
data = {
    "A": np.random.randint(1, 100, 10),
    "B": np.random.rand(10),
    "C": np.random.choice(['X', 'Y', 'Z'], 10)
}
df = pd.DataFrame(data)
print("Sample DataFrame created with NumPy:")
print(df)