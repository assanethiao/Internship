import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
data = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\indianpinearray.npy')
gt = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\IPgt.npy')

print("Image shape:", data.shape)
print("Ground truth shape:", gt.shape)

# Preprocessing
M, N, K = data.shape
X = data.reshape(M * N, K)
print("Dataset shape (pixels, bands):", X.shape)

# K-means clusturing
n_clusters = 16  # à tester
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
labels = kmeans.fit_predict(X)

# Reconstruction image
classification = labels.reshape(M, N)

# Visualisation
plt.figure(figsize=(6,5))
plt.imshow(classification, cmap='jet')
plt.title(f"K-means Classification - ({n_clusters} clusters)")
plt.axis('off')
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(gt, cmap='jet')
plt.title("Ground Truth")
plt.axis('off')
plt.show()
