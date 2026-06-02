import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load data
data = np.load('..\\dataset\\indianpinearray.npy')
gt = np.load('..\\dataset\\IPgt.npy')

# print("Image shape:", data.shape)
# print("Ground truth shape:", gt.shape)

# Preprocessing
M, N, K = data.shape
X = data.reshape(M * N, K)
# print("Dataset shape (pixels, bands):", X.shape)


def clustering_accuracy(gt, pred):

    # Images to vectors
    gt = gt.flatten()
    pred = pred.flatten()

    # Remove background (class 0)
    mask = gt != 0

    gt = gt[mask]
    pred = pred[mask]

    # Confuison matrix
    cm = confusion_matrix(gt, pred)

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Accuracy finale
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    return accuracy

'''
for n_clusters in range(2, 17):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    labels = kmeans.fit_predict(X_pca)

    # Reconstruction image
    classification = labels.reshape(M, N)

    accuracy = clustering_accuracy(gt, classification)
    print(f"Clustering Accuracy for {n_clusters} clusters: {accuracy:.4f}")


'''

# K-means clusturing
n_clusters =  7 # à tester
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

accuracy = clustering_accuracy(gt, classification)
print(f"Clustering Accuracy: {accuracy:.4f}")

'''

plt.figure(figsize=(6,5))
plt.imshow(gt, cmap='jet')
plt.title("Ground Truth")
plt.axis('off')
plt.show()

'''