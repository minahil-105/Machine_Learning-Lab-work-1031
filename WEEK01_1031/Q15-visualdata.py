# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import matplotlib.pyplot as plt
year = [2010, 2011, 2012, 2013, 2014, 2015]
population = [2.5, 2.7, 2.9, 3.2, 3.5, 3.7]
plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population in billions')
plt.title('Year vs Population')
plt.legend(['Population'])      
plt.show()