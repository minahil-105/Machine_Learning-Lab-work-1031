# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
from sklearn import datasets

#loading iris dataset
iris = datasets.load_iris()
print("Iris dataset loaded:")
print(iris.data[:5])  # Display the first five rows of the iris dataset


#for target values
print("\nTarget values:")
print(iris.target[:5])  # Display the first five target values of the iris dataset