# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd

# Load data from CSV file
df = pd.read_csv('network_metrics_reverse.csv')
print("DataFrame loaded from CSV:")
print(df.head())  # Display the first few rows of the DataFrame

#loading by the url
online_csv_url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv'
df_online = pd.read_csv(online_csv_url)
print("\nDataFrame loaded from online CSV:")
print(df_online.head())  # Display the first few rows of the DataFrame loaded from the