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
    """Normalise chaque bande spectrale pour avoir moyenne 0, écart-type 1."""
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

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        random_state=seed
    )
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
    Mesure combinée : produit de (1 - CE) et (1 - CIP).
    Interprétée comme une probabilité. On cherche à MAXIMISER.
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


# BOUCLE D'OPTIMISATION DES PARAMÈTRES c_k

def optimize_weights(data_norm, gt, n_clusters=16, n_iter=200, seed=0):
    """
    Recherche aléatoire des poids c_k (un par bande spectrale).
    À chaque itération :
      - on tire aléatoirement de nouveaux poids
      - on calcule la vraisemblance (CE + CIP)
      - on garde les poids qui maximisent la vraisemblance
    """
    rng = np.random.RandomState(seed)
    K   = data_norm.shape[2]

    # Initialisation : tous les poids à 1
    best_weights = np.ones(K)
    X0           = build_features(data_norm, best_weights)
    best_classif = run_kmeans(X0, n_clusters)
    best_score   = compute_vraisemblance(best_classif)

    history_ce      = [compute_CE(best_classif)]
    history_oac     = [clustering_accuracy(gt, best_classif)]
    history_score   = [best_score]

    print(f"Iter 000 | CE={history_ce[0]:.4f} | "
          f"OAC={history_oac[0]:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):

        # Tirage de nouveaux poids : valeurs positives entre 0 et 2
        new_weights = rng.uniform(0, 2, size=K)

        # Construction des features et clustering
        X_new    = build_features(data_norm, new_weights)
        classif  = run_kmeans(X_new, n_clusters, seed=t)
        score    = compute_vraisemblance(classif)

        # Mise à jour si amélioration
        if score > best_score:
            best_score   = score
            best_weights = new_weights
            best_classif = classif
            print(f"Iter {t:03d} | CE={compute_CE(classif):.4f} | "
                  f"OAC={clustering_accuracy(gt, classif):.4f} | "
                  f"score={score:.4f}  ✓ amélioration")

        history_ce.append(compute_CE(classif))
        history_oac.append(clustering_accuracy(gt, classif))
        history_score.append(score)

    return best_classif, best_weights, history_ce, history_oac, history_score


# LANCEMENT

n_clusters = 16

print("=== Optimisation en cours ===")
best_classif, best_weights, history_ce, history_oac, history_score = \
    optimize_weights(data_norm, gt, n_clusters=n_clusters, n_iter=200)

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

# Courbe CE vs OAC (relation vraisemblance / performance réelle)
plt.figure(figsize=(7, 5))
plt.scatter(history_ce, history_oac, alpha=0.4, s=10, c='steelblue')
plt.xlabel("CE (mesure de vraisemblance)")
plt.ylabel("OAC (performance réelle)")
plt.title("Relation CE ↔ OAC\n(tendance attendue : CE↓ → OAC↑)")
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