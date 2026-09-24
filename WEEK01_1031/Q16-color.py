# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import matplotlib.pyplot as plt
help(plt.plot)
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.hist(values, bins=5, color='blue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()