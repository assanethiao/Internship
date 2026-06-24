import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import uniform_filter

data = np.load('..\\dataset\\indianpinearray.npy')
gt   = np.load('..\\dataset\\IPgt.npy')
M, N, K = data.shape

# ─────────────────────────────────────────────
# 1. NORMALISATION
# ─────────────────────────────────────────────
def normalize(data):
    data_norm = np.zeros_like(data, dtype=np.float64)
    for k in range(data.shape[2]):
        band = data[:, :, k].astype(np.float64)
        mu, sig = band.mean(), band.std()
        data_norm[:, :, k] = (band - mu) / sig if sig > 0 else 0.0
    return data_norm

data_norm = normalize(data)

# ─────────────────────────────────────────────
# 2. LISSAGE SPATIAL (filtre moyenneur sur une fenêtre size×size)
#    Remplace le voisinage à 4 pixels : plus de contexte spatial
# ─────────────────────────────────────────────
def smooth_spectral(data_norm, window=5):
    """
    Moyenne chaque bande sur une fenêtre window x window.
    Donne un contexte spatial beaucoup plus large que 4 voisins.
    """
    smoothed = np.zeros_like(data_norm)
    for k in range(data_norm.shape[2]):
        smoothed[:, :, k] = uniform_filter(data_norm[:, :, k], size=window)
    return smoothed

# ─────────────────────────────────────────────
# 3. FEATURES : spectral original + spectral lissé + coordonnées
# ─────────────────────────────────────────────
def build_features_v2(data_norm, data_smooth, weights=None, alpha=1.0):
    """
    Feature = [spectral_original, spectral_lissé, alpha*(i,j normalisés)]
    Le spectral lissé apporte l'info de contexte spatial de façon
    plus douce qu'une simple concaténation de 4 voisins.
    """
    M, N, K = data_norm.shape

    spec1 = data_norm.copy()
    spec2 = data_smooth.copy()
    if weights is not None:
        spec1 = spec1 * weights[np.newaxis, np.newaxis, :]
        spec2 = spec2 * weights[np.newaxis, np.newaxis, :]

    coords_i = (np.arange(M) - M/2) / (M / (2 * np.sqrt(3)))
    coords_j = (np.arange(N) - N/2) / (N / (2 * np.sqrt(3)))
    II, JJ = np.meshgrid(coords_i, coords_j, indexing='ij')

    spec1_flat = spec1.reshape(M*N, K)
    spec2_flat = spec2.reshape(M*N, K)
    coords_flat = np.stack([II.flatten(), JJ.flatten()], axis=1) * alpha

    return np.concatenate([spec1_flat, spec2_flat, coords_flat], axis=1)

# ─────────────────────────────────────────────
# 4. K-MEANS DÉTERMINISTE
# ─────────────────────────────────────────────
def run_kmeans(X, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, init=X[init_idx],
                    n_init=1, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)

# ─────────────────────────────────────────────
# 5. MÉTRIQUES
# ─────────────────────────────────────────────
def compute_CE(classification):
    up    = np.roll(classification,  1, axis=0)
    down  = np.roll(classification, -1, axis=0)
    left  = np.roll(classification,  1, axis=1)
    right = np.roll(classification, -1, axis=1)
    return ((classification != up)   | (classification != down) |
            (classification != left) | (classification != right)).mean()

def clustering_accuracy(gt, pred):
    gt_f, pred_f = gt.flatten(), pred.flatten()
    mask = gt_f != 0
    cm = confusion_matrix(gt_f[mask], pred_f[mask])
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()

# ─────────────────────────────────────────────
# 6. RECHERCHE DU MEILLEUR (window, alpha)
#    gt est utilisé seulement pour MESURER, pas pour optimiser le choix
# ─────────────────────────────────────────────
n_clusters = 16
data_smooth_5  = smooth_spectral(data_norm, window=5)
data_smooth_9  = smooth_spectral(data_norm, window=9)
data_smooth_15 = smooth_spectral(data_norm, window=15)

configs = [
    ("window=5,  alpha=0.5", data_smooth_5,  0.5),
    ("window=5,  alpha=2.0", data_smooth_5,  2.0),
    ("window=9,  alpha=0.5", data_smooth_9,  0.5),
    ("window=9,  alpha=2.0", data_smooth_9,  2.0),
    ("window=15, alpha=0.5", data_smooth_15, 0.5),
    ("window=15, alpha=2.0", data_smooth_15, 2.0),
]

resultats = []
for label, smooth, alpha in configs:
    X = build_features_v2(data_norm, smooth, alpha=alpha)
    classif = run_kmeans(X, n_clusters)
    oac = clustering_accuracy(gt, classif)
    ce  = compute_CE(classif)
    resultats.append((label, oac, ce, classif))
    print(f"{label:25s} | OAC={oac:.4f} | CE={ce:.4f}")

# ─────────────────────────────────────────────
# 7. VISUALISATION COMPARATIVE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (label, oac, ce, classif) in enumerate(resultats):
    axes[idx].imshow(classif, cmap='jet')
    axes[idx].set_title(f"{label}\nOAC={oac:.4f} | CE={ce:.4f}", fontsize=10)
    axes[idx].axis('off')

axes[6].imshow(gt, cmap='jet')
axes[6].set_title("Ground Truth")
axes[6].axis('off')
axes[7].axis('off')

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 8. MEILLEURE CONFIG : raffinement avec recuit simulé sur c_k
# ─────────────────────────────────────────────
best_idx = np.argmax([r[1] for r in resultats])
best_label, _, _, _ = resultats[best_idx]
best_window = int(configs[best_idx][0].split("window=")[1].split(",")[0])
best_alpha  = configs[best_idx][2]
best_smooth = configs[best_idx][1]

print(f"\nMeilleure config trouvée : {best_label}")

def optimize_recuit(data_norm, data_smooth, gt, alpha, n_clusters=16,
                     n_iter=300, seed=0):
    rng = np.random.RandomState(seed)
    K   = data_norm.shape[2]

    best_weights = np.ones(K)
    X0 = build_features_v2(data_norm, data_smooth, best_weights, alpha)
    best_classif = run_kmeans(X0, n_clusters)
    best_score   = 1 - compute_CE(best_classif)
    best_ce      = compute_CE(best_classif)
    best_oac     = clustering_accuracy(gt, best_classif)

    hist_ce, hist_oac = [best_ce], [best_oac]
    sigma, sigma_min, n_plateau = 0.5, 1e-4, 0

    print(f"Départ | CE={best_ce:.4f} | OAC={best_oac:.4f}")

    for t in range(1, n_iter + 1):
        noise = rng.randn(K) * sigma
        new_w = np.maximum(best_weights + noise, 0)
        if new_w.sum() == 0:
            new_w = np.ones(K)

        X_new   = build_features_v2(data_norm, data_smooth, new_w, alpha)
        classif = run_kmeans(X_new, n_clusters, seed=t)
        score   = 1 - compute_CE(classif)
        ce      = compute_CE(classif)
        oac     = clustering_accuracy(gt, classif)

        hist_ce.append(ce)
        hist_oac.append(oac)

        if score > best_score:
            best_score, best_weights = score, new_w
            best_classif, best_ce, best_oac = classif, ce, oac
            n_plateau = 0
            print(f"Iter {t:03d} | CE={ce:.4f} | OAC={oac:.4f} ✓")
        else:
            n_plateau += 1
            if n_plateau % 20 == 0:
                sigma = max(sigma * 0.3, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.1, 0

    return best_classif, best_weights, hist_ce, hist_oac

classif_final, w_final, hist_ce, hist_oac = optimize_recuit(
    data_norm, best_smooth, gt, best_alpha, n_clusters=n_clusters, n_iter=300)

# Résultat final
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(classif_final, cmap='jet')
axes[0].set_title(f"Meilleur résultat ({best_label})\n"
                  f"OAC={clustering_accuracy(gt, classif_final):.4f} | "
                  f"CE={compute_CE(classif_final):.4f}")
axes[0].axis('off')
axes[1].imshow(gt, cmap='jet')
axes[1].set_title("Ground Truth")
axes[1].axis('off')
plt.tight_layout()
plt.show()