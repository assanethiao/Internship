import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

data = np.load('..\\dataset\\indianpinearray.npy')
gt   = np.load('..\\dataset\\IPgt.npy')
M, N, K = data.shape

def normalize(data):
    data_norm = np.zeros_like(data, dtype=np.float64)
    for k in range(data.shape[2]):
        band = data[:, :, k].astype(np.float64)
        mu, sig = band.mean(), band.std()
        data_norm[:, :, k] = (band - mu) / sig if sig > 0 else 0.0
    return data_norm

data_norm = normalize(data)

def build_features_spatial(data, weights=None, alpha=1.0):
    M, N, K = data.shape
    if weights is not None:
        data = data * weights[np.newaxis, np.newaxis, :]
    data_pad = np.pad(data, ((1,1),(1,1),(0,0)), mode="reflect")

    coords_i = (np.arange(M) - M/2) / (M / (2 * np.sqrt(3)))
    coords_j = (np.arange(N) - N/2) / (N / (2 * np.sqrt(3)))

    features = []
    for i in range(1, M+1):
        for j in range(1, N+1):
            spectral = np.concatenate([
                data_pad[i,   j,   :],
                data_pad[i-1, j,   :],
                data_pad[i+1, j,   :],
                data_pad[i,   j-1, :],
                data_pad[i,   j+1, :]
            ])
            spatial = np.array([alpha * coords_i[i-1],
                                 alpha * coords_j[j-1]])
            features.append(np.concatenate([spectral, spatial]))
    return np.array(features)

def run_kmeans(X, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, init=X[init_idx],
                    n_init=1, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)

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
# TEST SUR PLUSIEURS VALEURS D'ALPHA
# ─────────────────────────────────────────────
n_clusters = 16
alphas = [4.9, 5.0, 5.1, 5.2, 5.3, 5.5]

resultats = []
for alpha in alphas:
    X = build_features_spatial(data_norm, alpha=alpha)
    classif = run_kmeans(X, n_clusters)
    oac = clustering_accuracy(gt, classif)
    ce  = compute_CE(classif)
    resultats.append((alpha, oac, ce, classif))
    print(f"alpha={alpha:.1f} | OAC={oac:.4f} | CE={ce:.4f}")

# ─────────────────────────────────────────────
# VISUALISATION : une carte par valeur d'alpha
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (alpha, oac, ce, classif) in enumerate(resultats):
    axes[idx].imshow(classif, cmap='jet')
    axes[idx].set_title(f"alpha={alpha} | OAC={oac:.4f} | CE={ce:.4f}")
    axes[idx].axis('off')

axes[-1].imshow(gt, cmap='jet')
axes[-1].set_title("Ground Truth")
axes[-1].axis('off')

plt.suptitle("Effet du paramètre alpha (importance spatiale)", fontsize=14)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# RECUIT SIMULÉ avec le meilleur alpha trouvé
# ─────────────────────────────────────────────
best_alpha = alphas[np.argmax([r[1] for r in resultats])]
print(f"\nMeilleur alpha : {best_alpha}")

def optimize_recuit_spatial(data_norm, gt, alpha, n_clusters=16,
                             n_iter=500, seed=0):
    rng = np.random.RandomState(seed)
    K   = data_norm.shape[2]

    best_weights = np.ones(K)
    X0 = build_features_spatial(data_norm, best_weights, alpha=alpha)
    best_classif = run_kmeans(X0, n_clusters)
    best_score   = 1 - compute_CE(best_classif)  # on minimise CE
    best_oac     = clustering_accuracy(gt, best_classif)
    best_ce      = compute_CE(best_classif)

    hist_ce, hist_oac = [best_ce], [best_oac]

    sigma, sigma_min = 1.0, 0.00001
    n_plateau = 0

    print(f"Départ | CE={best_ce:.4f} | OAC={best_oac:.4f}")

    for t in range(1, n_iter + 1):
        noise = rng.randn(K) * sigma
        new_w = np.maximum(best_weights + noise, 0)
        if new_w.sum() == 0:
            new_w = np.ones(K)

        X_new   = build_features_spatial(data_norm, new_w, alpha=alpha)
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

classif_final, w_final, hist_ce, hist_oac = \
    optimize_recuit_spatial(data_norm, gt, alpha=best_alpha,
                            n_clusters=n_clusters, n_iter=500)

# Résultat final
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(classif_final, cmap='jet')
axes[0].set_title(f"Recuit simulé + spatial (alpha={best_alpha})\n"
                  f"OAC={clustering_accuracy(gt, classif_final):.4f} | "
                  f"CE={compute_CE(classif_final):.4f}")
axes[0].axis('off')
axes[1].imshow(gt, cmap='jet')
axes[1].set_title("Ground Truth")
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Courbe CE vs OAC
plt.figure(figsize=(7, 5))
plt.scatter(hist_ce, hist_oac, alpha=0.5, s=10, c='steelblue')
plt.xlabel("CE")
plt.ylabel("OAC")
plt.title(f"CE vs OAC (alpha={best_alpha})")
plt.grid(True)
plt.tight_layout()
plt.show()