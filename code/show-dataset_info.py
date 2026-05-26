import numpy as np
import matplotlib.pyplot as plt

data = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\indianpinearray.npy')  # Hyperspectral image 
gt = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\IPgt.npy') # Ground truth 

print("=== Indian Pines Dataset Info ===")
print(f"Data shape: {data.shape} (Height, Width, Spectral Bands)")
print(f"Ground truth shape: {gt.shape} (Height, Width)")
print(f"Number of spectral bands: {data.shape[2]}")
print(f"Image dimensions: {data.shape[0]} x {data.shape[1]}")
print(f"Unique classes in ground truth: {np.unique(gt)}")
print(f"Total pixels: {data.shape[0] * data.shape[1]}")

# Show a single spectral band as a grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(data[:, :, 30], cmap='gray')
plt.title('Spectral band 30')
plt.axis('off')
plt.show()

# Show the ground truth map
plt.figure(figsize=(6, 4))
plt.imshow(gt, cmap='jet')
plt.title('Ground Truth (Land Cover Classes)')
plt.axis('off')
plt.show()

row, col = 60, 80
plt.figure(figsize=(8, 4))
plt.plot(data[row, col, :])
plt.title(f'Spectral Signature at Pixel ({row}, {col})')
plt.xlabel('Band')
plt.ylabel('Reflectance')
plt.show()

