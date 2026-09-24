# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd

# Load data from json file
df = pd.read_json('data.json')
print("DataFrame loaded from JSON:")
print(df.head())  # Display the first few rows of the DataFrame

#loading from the json url
online_json_url = 'https://api.github.com/repos/pandas-dev/pandas/issues?per_page=5'
df_online_json = pd.read_json(online_json_url)
print("\nDataFrame loaded from online JSON:")
print(df_online_json.head())  # Display the first few rows of the DataFrame loaded from