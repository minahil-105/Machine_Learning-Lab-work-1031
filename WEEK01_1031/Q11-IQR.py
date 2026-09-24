# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import pandas as pd

data=pd.DataFrame({'income':[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,200000]})

Q1=data['income'].quantile(0.25)
Q3=data['income'].quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR

outliers=data[(data['income']<lower_limit) | (data['income']>upper_limit)]
print(outliers)
data_without_outliers=data[(data['income']>=lower_limit) & (data['income']<=upper_limit)]
print(data_without_outliers)