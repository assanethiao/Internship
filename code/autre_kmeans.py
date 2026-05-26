import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# LOAD DATA
data = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\indianpinearray.npy')
gt = np.load('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset\\IPgt.npy')

print("Image shape:", data.shape)
print("Ground truth shape:", gt.shape)


# PREPROCESSING
M, N, K = data.shape

# Transformation image - matrice
X = data.reshape(M * N, K)

print("Dataset shape:", X.shape)


# EUCLIDEAN DISTANCE
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


#  K-MEANS

class MyKMeans:

    def __init__(self, k=16, max_iters=10):
        self.k = k
        self.max_iters = max_iters

    # Initialisation aléatoire des centroïdes
    def initialize_centroids(self, X):

        centroids_idx = np.random.choice(X.shape[0],self.k,replace=False)

        return X[centroids_idx]

    # Attribution des clusters
    def assign_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.k)]
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = [
                euclidean_distance(x, centroid)
                for centroid in centroids
            ]

            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)
            labels[i] = cluster_idx

        return clusters, labels

    # Mise à jour des centroïdes
    def update_centroids(self, X, clusters):

        centroids = np.zeros((self.k, X.shape[1]))
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                centroids[i] = X[np.random.randint(0, X.shape[0])]

            else:
                centroids[i] = np.mean(X[cluster],axis=0)

        return centroids

    # Entraînement
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        history = []
        for iteration in range(self.max_iters):
            clusters, labels = self.assign_clusters(X,self.centroids)
            history.append((self.centroids.copy(),labels.copy()))

            prev_centroids = self.centroids.copy()
            self.centroids = self.update_centroids(X,clusters)

            # Condition d'arrêt
            if np.allclose(prev_centroids, self.centroids):

                print("Convergence atteinte")
                break

        return history


# TRAIN MODEL

kmeans = MyKMeans(k=16, max_iters=10)
history = kmeans.fit(X)


# FINAL CLASSIFICATION

final_centroids, final_labels = history[-1]
classification = final_labels.reshape(M, N)


# VISUALISATION

plt.figure(figsize=(6,5))
plt.imshow(classification, cmap='jet')
plt.title("Custom K-Means Classification")
plt.axis('off')
plt.show()


# GROUND TRUTH

plt.figure(figsize=(6,5))
plt.imshow(gt, cmap='jet')
plt.title("Ground Truth")
plt.axis('off')
plt.show()