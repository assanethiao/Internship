import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

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
# 2. FEATURES SPATIALES (spectral + voisinage + coordonnées * alpha)
# ─────────────────────────────────────────────
def build_features_spatial(data, weights=None, alpha=1.0):
    M, N, K = data.shape
    if weights is not None:
        data = data * weights[np.newaxis, np.newaxis, :]
    data_pad = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="reflect")

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


# ─────────────────────────────────────────────
# 3. MÉTRIQUES (CE et CIP ensemble, en une seule passe)
# ─────────────────────────────────────────────
def compute_CE_CIP(classification):
    up    = np.roll(classification,  1, axis=0)
    down  = np.roll(classification, -1, axis=0)
    left  = np.roll(classification,  1, axis=1)
    right = np.roll(classification, -1, axis=1)
    ce = ((classification != up)   | (classification != down) |
          (classification != left) | (classification != right)).mean()

    pad = np.pad(classification, 1, mode='edge')
    isolated = np.ones((M, N), dtype=bool)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor = pad[1+di:M+1+di, 1+dj:N+1+dj]
            isolated &= (classification != neighbor)
    cip = isolated.mean()
    return ce, cip


def clustering_accuracy(gt, pred):
    gt_f, pred_f = gt.flatten(), pred.flatten()
    mask = gt_f != 0
    cm = confusion_matrix(gt_f[mask], pred_f[mask])
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()


# ─────────────────────────────────────────────
# 4. CALIBRAGE DU SCORE (équation 1 du document du 19 juin)
#    Refait pour chaque alpha car l'échelle des features change
# ─────────────────────────────────────────────
def calibrate_mu_sigma(alpha, n_clusters=16, n_samples=20, seed=999):
    rng = np.random.RandomState(seed)
    ce_samples, cip_samples = [], []
    for s in range(n_samples):
        weights = rng.uniform(0, 2, size=K)
        X = build_features_spatial(data_norm, weights, alpha=alpha)
        classif = run_kmeans(X, n_clusters, seed=s)
        ce, cip = compute_CE_CIP(classif)
        ce_samples.append(ce)
        cip_samples.append(cip)
    mu_ce, sigma_ce   = np.mean(ce_samples),  np.std(ce_samples)
    mu_cip, sigma_cip = np.mean(cip_samples), np.std(cip_samples)
    return mu_ce, sigma_ce, mu_cip, sigma_cip


def compute_score(ce, cip, calibration):
    mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
    term_ce  = (mu_ce  - ce)  / sigma_ce  if sigma_ce  > 0 else 0
    term_cip = (mu_cip - cip) / sigma_cip if sigma_cip > 0 else 0
    return 0.5 * (term_ce + term_cip)


# ─────────────────────────────────────────────
# 5. RECUIT SIMULÉ À DIRECTION PRIVILÉGIÉE + SPATIAL (alpha)
# ─────────────────────────────────────────────
def optimize_directionnel_spatial(data_norm, gt, alpha, calibration,
                                    n_clusters=16, n_iter=500,
                                    seed=0, poids_direction=0.5):
    """
    Combine :
      - les features spatiales (alpha) qui donnaient le meilleur résultat
        visuel jusqu'ici
      - le recuit simulé à direction privilégiée (équation 3 du document
        du 19 juin), qui corrige l'exploration purement aléatoire
      - le score normalisé équilibrant CE et CIP
    """
    rng = np.random.RandomState(seed)
    K = data_norm.shape[2]

    best_weights = np.ones(K)
    X0 = build_features_spatial(data_norm, best_weights, alpha=alpha)
    best_classif = run_kmeans(X0, n_clusters)
    ce0, cip0 = compute_CE_CIP(best_classif)
    best_score = compute_score(ce0, cip0, calibration)
    best_ce, best_cip = ce0, cip0
    best_oac = clustering_accuracy(gt, best_classif)

    delta_x = rng.randn(K)
    delta_x /= np.linalg.norm(delta_x)

    hist_ce, hist_cip, hist_oac, hist_score, hist_sigma = (
        [best_ce], [best_cip], [best_oac], [best_score], [])

    sigma_init, sigma_min = 0.5, 0.001
    sigma = sigma_init
    n_plateau = 0

    print(f"[alpha={alpha}] Départ | CE={best_ce:.4f} | CIP={best_cip:.4f} "
          f"| OAC={best_oac:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):
        b_prime_t = rng.randn()
        b_t = rng.randn(K)
        noise = sigma * (poids_direction * b_prime_t * delta_x
                          + (1 - poids_direction) * b_t)

        new_weights = np.maximum(best_weights + noise, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        X_new   = build_features_spatial(data_norm, new_weights, alpha=alpha)
        classif = run_kmeans(X_new, n_clusters, seed=t)
        ce, cip = compute_CE_CIP(classif)
        score   = compute_score(ce, cip, calibration)
        oac     = clustering_accuracy(gt, classif)

        hist_ce.append(ce)
        hist_cip.append(cip)
        hist_oac.append(oac)
        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            delta_x_new = new_weights - best_weights
            norme = np.linalg.norm(delta_x_new)
            if norme > 1e-12:
                delta_x = delta_x_new / norme

            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip, best_oac = classif, ce, cip, oac
            n_plateau = 0
            print(f"[alpha={alpha}] Iter {t:04d} | CE={ce:.4f} | "
                  f"CIP={cip:.4f} | OAC={oac:.4f} | score={score:.4f} ✓")
        else:
            n_plateau += 1
            if n_plateau % 20 == 0:
                sigma = max(sigma * 0.3, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = sigma_init * 0.3, 0

    print(f"[alpha={alpha}] Final | OAC={best_oac:.4f} | CE={best_ce:.4f} | "
          f"CIP={best_cip:.4f}\n")

    return {
        "alpha": alpha, "classif": best_classif, "weights": best_weights,
        "hist_ce": hist_ce, "hist_cip": hist_cip, "hist_oac": hist_oac,
        "hist_score": hist_score, "hist_sigma": hist_sigma,
        "final_oac": best_oac, "final_ce": best_ce, "final_cip": best_cip,
    }


# ─────────────────────────────────────────────
# 6. BALAYAGE D'ALPHA AVEC LA MÉTHODE COMBINÉE
# ─────────────────────────────────────────────
n_clusters = 16
alphas = [0.5, 1.0, 2.0, 5.0, 8.0]

resultats = []
for alpha in alphas:
    calibration = calibrate_mu_sigma(alpha, n_clusters=n_clusters, n_samples=20)
    res = optimize_directionnel_spatial(
        data_norm, gt, alpha=alpha, calibration=calibration,
        n_clusters=n_clusters, n_iter=400, poids_direction=0.5
    )
    resultats.append(res)

# ─────────────────────────────────────────────
# 7. VISUALISATION COMPARATIVE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, res in enumerate(resultats):
    axes[idx].imshow(res["classif"], cmap='jet')
    axes[idx].set_title(f"alpha={res['alpha']} | OAC={res['final_oac']:.4f} | "
                        f"CE={res['final_ce']:.4f}")
    axes[idx].axis('off')

axes[-1].imshow(gt, cmap='jet')
axes[-1].set_title("Ground Truth")
axes[-1].axis('off')

plt.suptitle("Direction privilégiée + spatial : effet de alpha", fontsize=14)
plt.tight_layout()
plt.show()

# Tableau récapitulatif
print("\n=== Tableau récapitulatif ===")
print(f"{'alpha':>8} | {'OAC':>8} | {'CE':>8} | {'CIP':>8}")
print("-" * 42)
for res in resultats:
    print(f"{res['alpha']:>8.2f} | {res['final_oac']:>8.4f} | "
          f"{res['final_ce']:>8.4f} | {res['final_cip']:>8.4f}")

# ─────────────────────────────────────────────
# 8. MEILLEUR RÉSULTAT EN DÉTAIL
# ─────────────────────────────────────────────
best_res = max(resultats, key=lambda r: r["final_oac"])
print(f"\nMeilleur alpha : {best_res['alpha']}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(best_res["classif"], cmap='jet')
axes[0].set_title(f"Meilleur résultat (alpha={best_res['alpha']})\n"
                  f"OAC={best_res['final_oac']:.4f} | CE={best_res['final_ce']:.4f}")
axes[0].axis('off')
axes[1].imshow(gt, cmap='jet')
axes[1].set_title("Ground Truth")
axes[1].axis('off')
plt.tight_layout()
plt.show()

# CE, CIP, sigma au cours des itérations (Figure 1 du document du 19 juin)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(best_res["hist_cip"], color='darkorange')
axes[0].set_title("Évolution de CIP")
axes[0].grid(True)
axes[1].plot(best_res["hist_ce"], color='steelblue')
axes[1].set_title("Évolution de CE")
axes[1].grid(True)
axes[2].plot(best_res["hist_sigma"], color='purple')
axes[2].set_title("Évolution de sigma")
axes[2].grid(True)
plt.tight_layout()
plt.show()

# Score vs OAC
plt.figure(figsize=(7, 5))
plt.scatter(best_res["hist_score"], best_res["hist_oac"],
           alpha=0.4, s=10, c='steelblue')
plt.xlabel("Score normalisé")
plt.ylabel("OAC")
plt.title(f"Score vs OAC (alpha={best_res['alpha']})")
plt.grid(True)
plt.tight_layout()
plt.show()