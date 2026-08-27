import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# CHARGEMENT ET NORMALISATION

def load_data(data_path, gt_path):
    data = np.load(data_path)
    gt   = np.load(gt_path)
    return data, gt


def normalize(data):
    data_norm = np.zeros_like(data, dtype=np.float64)
    for k in range(data.shape[2]):
        band = data[:, :, k].astype(np.float64)
        mu, sig = band.mean(), band.std()
        data_norm[:, :, k] = (band - mu) / sig if sig > 0 else 0.0
    return data_norm


# FEATURES

def build_features(data, weights=None, alpha=0.0):
    """
    Pour chaque pixel : concatène [centre, haut, bas, gauche, droite]
    """
    M, N, K = data.shape
    if weights is not None:
        data = data * weights[np.newaxis, np.newaxis, :]
    data_pad = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="reflect")

    features = []
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            spectral = np.concatenate([
                data_pad[i,   j,   :],
                data_pad[i-1, j,   :],
                data_pad[i+1, j,   :],
                data_pad[i,   j-1, :],
                data_pad[i,   j+1, :]
            ])
            features.append(spectral)
    return np.array(features)


def run_kmeans(X, n_clusters, M, N, seed=42):
    
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, init=X[init_idx],
                    n_init=1, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)


# MÉTRIQUES

def compute_CE_CIP(classification):
    M, N = classification.shape
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


# CALIBRAGE DU SCORE 

def calibrate_score(data_norm, n_clusters, n_samples=30, seed=42, alpha=0.0):
    """
    Estime mu_CE, sigma_CE, mu_CIP, sigma_CIP par tirages aléatoires
    """
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    ce_samples, cip_samples = [], []
    for s in range(n_samples):
        weights = rng.uniform(0, 2, size=K)
        X = build_features(data_norm, weights)
        classif = run_kmeans(X, n_clusters, M, N, seed=42)
        ce, cip = compute_CE_CIP(classif)
        ce_samples.append(ce)
        cip_samples.append(cip)

    mu_ce,  sigma_ce  = np.mean(ce_samples),  np.std(ce_samples)
    mu_cip, sigma_cip = np.mean(cip_samples), np.std(cip_samples)

    print(f"Calibration : mu_CE={mu_ce:.4f}, sigma_CE={sigma_ce:.4f} | "
          f"mu_CIP={mu_cip:.4f}, sigma_CIP={sigma_cip:.4f}")

    return mu_ce, sigma_ce, mu_cip, sigma_cip


def compute_score_normalized(ce, cip, mu_ce, sigma_ce, mu_cip, sigma_cip):
    term_ce  = (mu_ce  - ce)  / sigma_ce  if sigma_ce  > 0 else 0
    term_cip = (mu_cip - cip) / sigma_cip if sigma_cip > 0 else 0
    return 0.5 * (term_ce + term_cip)


# RECUIT SIMULÉ À DIRECTION PRIVILÉGIÉE 
def optimize_weights_directional(data_norm, gt, n_clusters=16, n_iter=300, seed=0, w_direction=0.5,
                                  alpha=0.0, calibration=None,
                                  verbose=True):
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    def score_fn(classif):
        ce, cip = compute_CE_CIP(classif)
        oac = clustering_accuracy(gt, classif)
        if calibration is not None:
            mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
            normalized_score = compute_score_normalized(ce, cip, mu_ce, sigma_ce,
                                                       mu_cip, sigma_cip)
            score = 0.3 * normalized_score + 0.7 * oac
        else:
            score = 0.5 * ((1 - ce) * (1 - cip)) + 0.5 * oac
        return score, ce, cip, oac

    best_weights = np.ones(K)
    X0 = build_features(data_norm, best_weights, alpha=alpha)
    best_classif = run_kmeans(X0, n_clusters, M, N)
    best_score, best_ce, best_cip, best_oac = score_fn(best_classif)

    delta_x = rng.randn(K)
    delta_x /= np.linalg.norm(delta_x)

    hist_ce, hist_cip, hist_oac, hist_score, hist_sigma = (
        [best_ce], [best_cip], [best_oac], [best_score], [])

    sigma, sigma_min = 0.8, 1e-4
    n_plateau = 0

    if verbose:
        print(f"[alpha={alpha}] Départ | CE={best_ce:.4f} | CIP={best_cip:.4f} "
              f"| OAC={best_oac:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):
        b_prime = rng.randn()
        direction_term = b_prime * delta_x
        b_t = rng.randn(K)
        noise = sigma * (w_direction * direction_term + (1 - w_direction) * b_t)

        new_weights = np.maximum(best_weights + noise, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        X_new = build_features(data_norm, new_weights, alpha=alpha)
        classif = run_kmeans(X_new, n_clusters, M, N, seed=42)
        score, ce, cip, oac = score_fn(classif)

        hist_ce.append(ce)
        hist_cip.append(cip)
        hist_oac.append(oac)
        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            delta_x_new = new_weights - best_weights
            norm = np.linalg.norm(delta_x_new)
            if norm > 1e-12:
                delta_x = delta_x_new / norm
            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip, best_oac = classif, ce, cip, oac
            n_plateau = 0
            if verbose:
                print(f"[alpha={alpha}] Iter {t:04d} | CE={ce:.4f} | "
                      f"CIP={cip:.4f} | OAC={oac:.4f} | score={score:.4f} ✓")
        else:
            n_plateau += 1
            if n_plateau % 50 == 0:
                sigma = max(sigma * 0.9, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.1, 0

    if verbose:
        print(f"[alpha={alpha}] Final | OAC={best_oac:.4f} | CE={best_ce:.4f} | "
              f"CIP={best_cip:.4f} | score={best_score:.4f}\n")

    return {
        "alpha": alpha,
        "classif": best_classif,
        "weights": best_weights,
        "hist_ce": hist_ce,
        "hist_cip": hist_cip,
        "hist_oac": hist_oac,
        "hist_score": hist_score,
        "hist_sigma": hist_sigma,
        "final_oac": best_oac,
        "final_ce": best_ce,
    }


# on lance l'optimisation complète pour chaque alpha

def sweep_alpha(data_norm, gt, alphas, n_clusters=16, n_iter=300,
                 w_direction=0.999, n_iter_calib=30):
    """
    Pour chaque valeur d'alpha :
      calibre mu/sigma (alpha modifie l'échelle des features, donc
         la calibration doit être refaite pour chaque alpha)
      lance le recuit simulé à direction privilégiée
      garde le résultat
    """
    resultats = []
    for alpha in alphas:
        print(f"\n========== ALPHA = {alpha} ==========")
        calibration = calibrate_score(data_norm, n_clusters, alpha=alpha,
                                       n_samples=n_iter_calib)
        result = optimize_weights_directional(
            data_norm, gt, n_clusters=n_clusters, n_iter=n_iter,
            w_direction=w_direction, alpha=alpha, calibration=calibration,
            verbose=True
        )
        resultats.append(result)
    return resultats


# VISUALISATION
def plot_best_result(best_result, gt):
    classif = best_result["classif"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(classif, cmap='jet')
    axes[0].set_title(f"Meilleur résultat (alpha={best_result['alpha']})\n" f"OAC={best_result['final_oac']:.4f} | " f"CE={best_result['final_ce']:.4f}")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(best_result["hist_cip"], color='darkorange')
    axes[0].set_title("Évolution de CIP")
    axes[0].grid(True)
    axes[1].plot(best_result["hist_ce"], color='steelblue')
    axes[1].set_title("Évolution de CE")
    axes[1].grid(True)
    axes[2].plot(best_result["hist_sigma"], color='purple')
    axes[2].set_title("Évolution de sigma")
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_hungarian(gt, classif, n_classes=16):

    # Ignorer le fond (classe 0)
    mask = gt.flatten() != 0

    y_true = gt.flatten()[mask]
    y_pred = classif.flatten()[mask]

    cm = confusion_matrix(y_true, y_pred)

    # Appariement optimal avec l'algorithme hongrois
    rows, cols = linear_sum_assignment(-cm)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(cm)

    # Afficher les valeurs
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    # Entourer les correspondances optimales
    for r, c in zip(rows, cols):
        ax.plot(
            c, r,
            marker='o',
            markersize=18,
            markerfacecolor='none',
            markeredgewidth=2
        )

    # Axes
    ax.set_xlabel("Clusters K-Means")
    ax.set_ylabel("Classes Ground Truth")

    ax.set_title(
        "Matrice de confusion et appariement optimal\n"
        "Indian Pines"
    )

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    ax.set_xticklabels(
        [f"C{i}" for i in range(cm.shape[1])]
    )

    ax.set_yticklabels(
        [f"Classe {i+1}" for i in range(cm.shape[0])]
    )

    plt.tight_layout()

    # Sauvegarder la figure
    plt.savefig(
        "figure_2_2_matrice_confusion.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Afficher l'appariement dans la console
    print("\n=== APPARIEMENT OPTIMAL ===")

    for r, c in zip(rows, cols):
        print(
            f"Cluster C{c} -> Classe {r+1} "
            f"({cm[r, c]} pixels)"
        )



# POINT D'ENTRÉE

if __name__ == "__main__":
    data, gt = load_data('..\\dataset\\indianpinearray.npy', '..\\dataset\\IPgt.npy')
    data_norm = normalize(data)
    n_clusters = 16

    # Balayage : on teste plusieurs alpha, chacun avec sa propre
    # optimisation complète par direction privilégiée
    alphas = [8]

    resultats = sweep_alpha(data_norm, gt, alphas, n_clusters=n_clusters,
                             n_iter=200, w_direction=0.99, n_iter_calib=30)

    # Sélection du meilleur alpha (selon OAC, juste pour visualiser le résultat)
    best_result = max(resultats, key=lambda r: r["final_oac"])
    print(f"\nMeilleur alpha trouvé : {best_result['alpha']}")

    plot_best_result(best_result, gt)
    plot_confusion_hungarian(gt, best_result["classif"], n_classes=16)