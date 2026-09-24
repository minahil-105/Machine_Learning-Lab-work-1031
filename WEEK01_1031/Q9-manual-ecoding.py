# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR

from turtle import pd


mapping = { 
    'h' :1 , 'u' :2 , 's' :3 , 'n' :4 }
features = pd.DataFrame({'type': ['h', 'u', 's', 'n', 'a']})
features['type'] = features['type'].map(mapping)
print(features.groupby('type').size()) 