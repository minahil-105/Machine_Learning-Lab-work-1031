# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
from sklearn.preprocessing import LabelEncoder
import pandas as pd

features = pd.DataFrame({'type': ['h', 'u', 's', 'n', 'a']})

le = LabelEncoder()
features['type'] = le.fit_transform(features['type'])
print ("label encoder type")
print(features.values)