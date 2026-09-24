# 23-NTU-CS-1031
# HAFIZA MINAHIL SHABBIR
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
# Load the image
image = plt.imread('bird.jpg')
# Display the image
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(image)
    plt.axis('off')

plt.show()