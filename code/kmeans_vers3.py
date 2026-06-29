import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load data
data = np.load('..\\dataset\\indianpinearray.npy')
gt   = np.load('..\\dataset\\IPgt.npy')

M, N, K = data.shape

# Data normalization (per band)

def normalize(data):
    """Normalise chaque bande spectrale pour avoir moyenne 0, écart-type 1 """
    data_norm = np.zeros_like(data, dtype=np.float64)
    for k in range(data.shape[2]):
        band = data[:, :, k].astype(np.float64)
        mu   = band.mean()
        sig  = band.std()
        if sig > 0:
            data_norm[:, :, k] = (band - mu) / sig
        else:
            data_norm[:, :, k] = 0.0
    return data_norm

data_norm = normalize(data)


# Build features

def build_features(data, weights=None):
    """
    Pour chaque pixel : concatène [centre, haut, bas, gauche, droite].
    weights : vecteur de longueur K pour pondérer les bandes (paramètres c_k).
              Si None, toutes les bandes ont le même poids.
    """
    M, N, K = data.shape

    if weights is not None:
        # Applique c_k à chaque bande
        data = data * weights[np.newaxis, np.newaxis, :]

    data_pad = np.pad(data, ((1,1),(1,1),(0,0)), mode="reflect")
    features = []

    for i in range(1, M+1):
        for j in range(1, N+1):
            centre = data_pad[i,   j,   :]
            haut   = data_pad[i-1, j,   :]
            bas    = data_pad[i+1, j,   :]
            gauche = data_pad[i,   j-1, :]
            droite = data_pad[i,   j+1, :]
            features.append(np.concatenate([centre, haut, bas, gauche, droite]))

    return np.array(features)


# K-MEANS (déterministe via init fixe)

def run_kmeans(X, n_clusters, seed=42):

    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    init_centers = X[init_idx]

    kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1, random_state=seed)
    labels = kmeans.fit_predict(X)
    return labels.reshape(M, N)


# MESURES DE VRAISEMBLANCE

def compute_CE(classification):
    """
    CE : proportion de pixels dont au moins un voisin (4-connexe)
    appartient à une classe différente.
    Valeur entre 0 et 1. On cherche à MINIMISER CE.
    """
    # Décalages dans les 4 directions
    up    = np.roll(classification,  1, axis=0)
    down  = np.roll(classification, -1, axis=0)
    left  = np.roll(classification,  1, axis=1)
    right = np.roll(classification, -1, axis=1)

    # Un pixel est sur un contour si au moins un voisin diffère
    edge = (
        (classification != up)   |
        (classification != down) |
        (classification != left) |
        (classification != right)
    )
    return edge.mean() 


def compute_CIP(classification):
    """
    CIP : proportion de pixels isolés (tous les 8 voisins dans une autre classe).
    Valeur entre 0 et 1. On cherche à MINIMISER CIP.
    """
    pad = np.pad(classification, 1, mode='edge')
    isolated = np.ones((M, N), dtype=bool)

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor = pad[1+di:M+1+di, 1+dj:N+1+dj]
            isolated &= (classification != neighbor)

    return isolated.mean()


def compute_vraisemblance(classification):
    """
    Mesure combinée : produit de (1 - CE) et (1 - CIP). Interprétée comme une probabilité. On cherche à MAXIMISER.
    """
    ce  = compute_CE(classification)
    cip = compute_CIP(classification)
    return (1 - ce) * (1 - cip)


# MÉTRIQUE OAC (accuracy clustering)

def clustering_accuracy(gt, pred):
    gt_flat   = gt.flatten()
    pred_flat = pred.flatten()
    mask      = gt_flat != 0
    gt_m      = gt_flat[mask]
    pred_m    = pred_flat[mask]

    cm = confusion_matrix(gt_m, pred_m)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


# RECUIT SIMULÉ pour optimiser les poids c_k

def optimize_weights_recuit(data_norm, gt, n_clusters=16, n_iter=500, seed=0):
    """
    Recuit simulé pour optimiser les poids spectraux c_k.

    Idée : au lieu de tirer des poids complètement aléatoires,
    on explore AUTOUR des meilleurs poids trouvés, avec un pas
    qui diminue progressivement (sigma décroissant).
    """
    rng = np.random.RandomState(seed)
    K   = data_norm.shape[2]

    # Initialisation : poids uniformes
    best_weights = np.ones(K)
    X0 = build_features(data_norm, best_weights)
    best_classif = run_kmeans(X0, n_clusters)
    best_score   = compute_vraisemblance(best_classif)
    best_ce      = compute_CE(best_classif)
    best_oac     = clustering_accuracy(gt, best_classif)

    # Historiques
    hist_ce    = [best_ce]
    hist_oac   = [best_oac]
    hist_score = [best_score]
    hist_sigma = []

    # Paramètres du recuit
    sigma_init = 0.5   # écart-type initial pour l'exploration
    sigma_min  = 0.001    # écart-type minimal
    n_plateau  = 0        # compteur d'itérations sans amélioration
    sigma      = sigma_init

    print(f"Iter 000 | CE={best_ce:.4f} | OAC={best_oac:.4f} " f"| score={best_score:.4f} | sigma={sigma:.4f}")

    for t in range(1, n_iter + 1):

        # Exploration autour des meilleurs poids
        # avec un bruit gaussien de sigma décroissant
        noise       = rng.randn(K) * sigma
        new_weights = best_weights + noise
        new_weights = np.maximum(new_weights, 0)  # pas de poids négatifs

        # Fallback si tous les poids sont nuls
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        X_new   = build_features(data_norm, new_weights)
        classif = run_kmeans(X_new, n_clusters, seed=t)
        score   = compute_vraisemblance(classif)
        ce      = compute_CE(classif)
        oac     = clustering_accuracy(gt, classif)

        hist_ce.append(ce)
        hist_oac.append(oac)
        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            best_score   = score
            best_weights = new_weights
            best_classif = classif
            best_ce      = ce
            best_oac     = oac
            n_plateau    = 0
            print(f"Iter {t:03d} | CE={ce:.4f} | OAC={oac:.4f} "
                  f"| score={score:.4f} | sigma={sigma:.4f}  optimisé")
        else:
            n_plateau += 1

        # Réduction de sigma tous les 20 pas sans amélioration
        if n_plateau > 0 and n_plateau % 20 == 0:
            sigma = max(sigma * 0.8, sigma_min)

        # Redémarrage si sigma trop petit (exploration épuisée)
        if sigma <= sigma_min:
            sigma     = sigma_init * 0.3
            n_plateau = 0

    print(f"\n=== Résultat final ===")
    print(f"OAC = {best_oac:.4f} | CE = {best_ce:.4f} | " f"score = {best_score:.4f}")

    return best_classif, best_weights, hist_ce, hist_oac, hist_score, hist_sigma



# LANCEMENT

n_clusters = 16
n_iter     = 200

print("=== Optimisation en cours ===")
best_classif, best_weights, history_ce, history_oac, history_score, history_sigma = optimize_weights_recuit(data_norm, gt, n_clusters=n_clusters, n_iter=n_iter, seed=0)

final_oac = clustering_accuracy(gt, best_classif)
final_ce  = compute_CE(best_classif)
print(f"\n=== Résultat final ===")
print(f"OAC = {final_oac:.4f} | CE = {final_ce:.4f}")


# VISUALISATIONS

# Cartes de classification
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.imshow(best_classif, cmap='jet')
plt.title(f"Meilleure classification\nOAC={final_oac:.4f} | CE={final_ce:.4f}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gt, cmap='jet')
plt.title("Ground Truth")
plt.axis('off')
plt.tight_layout()
plt.show()

# Courbe CE vs OAC
plt.figure(figsize=(7, 5))
plt.scatter(history_ce, history_oac, alpha=0.4, s=10, c='steelblue')
plt.xlabel("CE (mesure de vraisemblance)")
plt.ylabel("OAC (performance réelle)")
plt.title("Relation CE - OAC")
plt.grid(True)
plt.tight_layout()
plt.show()

# Évolution du score combiné
plt.figure(figsize=(7, 4))
plt.plot(history_score, color='darkorange')
plt.xlabel("Itération")
plt.ylabel("Score vraisemblance (1-CE)×(1-CIP)")
plt.title("Évolution du score de vraisemblance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisation des poids spectraux optimisés c_k

plt.figure(figsize=(12,4))
plt.plot(best_weights, linewidth=2)
plt.xlabel("Indice de bande spectrale k")
plt.ylabel("Poids c_k")
plt.title("Poids spectraux optimisés")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisation des bandes les plus importantes (c_k > 1)
plt.figure(figsize=(12,4))
plt.bar(np.arange(K), best_weights)
plt.xlabel("Indice de bande spectrale k")
plt.ylabel("Poids c_k")
plt.title("Importance des bandes spectrales")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Visualisation vraisemblance vs OAC

plt.figure(figsize=(7, 5))
plt.scatter(history_score, history_oac, alpha=0.4, s=10, c='steelblue')
plt.xlabel("Score vraisemblance")
plt.ylabel("OAC (performance réelle)")
plt.title("Relation Score - OAC")
plt.grid(True)
plt.tight_layout()
plt.show()