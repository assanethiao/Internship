import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load data
data = np.load('..\\dataset\\indianpinearray.npy')
gt = np.load('..\\dataset\\IPgt.npy')

# print("Image shape:", data.shape) # print("Ground truth shape:", gt.shape) # X = data.reshape(M * N, K) # print("Dataset shape (pixels, bands):", X.shape)

# Preprocessing
M, N, K = data.shape

data_pad = np.pad(data, ((1,1),(1,1),(0,0)), mode='reflect')

'''
new_data = np.copy(data)

for i in range(1, M-1):
    for j in range(1, N-1):

        new_data[i,j,:] = (
            data[i,j,:]      # centre
            + data[i-1,j,:]  # haut
            + data[i+1,j,:]  # bas
            + data[i,j-1,:]  # gauche
            + data[i,j+1,:]  # droite
        ) / 5

X = new_data.reshape(M * N, K)
'''

features = []

for i in range(1, M+1):
    for j in range(1, N+1):


        centre = data_pad[i,j,:]
        haut   = data_pad[i-1,j,:]
        bas    = data_pad[i+1,j,:]
        gauche = data_pad[i,j-1,:]
        droite = data_pad[i,j+1,:]


        centre = data_pad[i,j,:]
        haut   = data_pad[i-1,j,:]
        bas    = data_pad[i+1,j,:]
        gauche = data_pad[i,j-1,:]
        droite = data_pad[i,j+1,:]
        feature = np.concatenate([centre, haut, bas, gauche, droite])
        features.append(feature)

X = np.array(features)


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
n_clusters =  15 # à tester
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
labels = kmeans.fit_predict(X)

# Reconstruction image
classification = labels.reshape(M, N)

# Visualisation
plt.figure(figsize=(14,6))

# Classification
plt.subplot(1, 2, 1)
plt.imshow(classification, cmap='jet')
plt.title(f"K-means Classification ({n_clusters} clusters)")
plt.axis('off')

# Ground Truth
plt.subplot(1, 2, 2)
plt.imshow(gt, cmap='jet')
plt.title("Ground Truth")
plt.axis('off')

plt.tight_layout()
plt.show()

accuracy = clustering_accuracy(gt, classification)
print(f"Clustering Accuracy: {accuracy:.4f}")

