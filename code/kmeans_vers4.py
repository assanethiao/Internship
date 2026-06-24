import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


# CHARGEMENT ET NORMALISATION (pas de variables globales implicites)
def load_data(data_path, gt_path):
    data = np.load(data_path)
    gt   = np.load(gt_path)
    return data, gt


def normalize(data):
    """Normalise chaque bande spectrale : moyenne 0, écart-type 1"""
    data_norm = np.zeros_like(data, dtype=np.float64)
    for k in range(data.shape[2]):
        band = data[:, :, k].astype(np.float64)
        mu, sig = band.mean(), band.std()
        data_norm[:, :, k] = (band - mu) / sig if sig > 0 else 0.0
    return data_norm


# FEATURES
def build_features(data, weights=None):
    """Pour chaque pixel : concatène [centre, haut, bas, gauche, droite]"""
    M, N, K = data.shape
    if weights is not None:
        data = data * weights[np.newaxis, np.newaxis, :]
    data_pad = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    features = []
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            features.append(np.concatenate([
                data_pad[i,   j,   :],
                data_pad[i-1, j,   :],
                data_pad[i+1, j,   :],
                data_pad[i,   j-1, :],
                data_pad[i,   j+1, :]
            ]))
    return np.array(features)


def run_kmeans(X, n_clusters, M, N, seed=42):
    """K-means déterministe """
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, init=X[init_idx],
                    n_init=1, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)


# MÉTRIQUES (CE et CIP calculés une seule fois, ensemble)
def compute_CE_CIP(classification):
    """
    Calcule CE et CIP en une seule passe pour éviter les calculs redondants
    Retourne (ce, cip)
    """
    M, N = classification.shape

    up    = np.roll(classification,  1, axis=0)
    down  = np.roll(classification, -1, axis=0)
    left  = np.roll(classification,  1, axis=1)
    right = np.roll(classification, -1, axis=1)

    diff_4 = ((classification != up)   | (classification != down) |
              (classification != left) | (classification != right))
    ce = diff_4.mean()

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


# CALIBRAGE DU SCORE (mu_i, sigma_i estimés par tirage aléatoire pur)
def calibrate_score(data_norm, n_clusters, n_samples=30, seed=999):
    """
    Estime mu_CE, sigma_CE, mu_CIP, sigma_CIP par tirages aléatoires
    """
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    ce_samples, cip_samples = [], []
    for s in range(n_samples):
        weights = rng.uniform(0, 2, size=K)
        X = build_features(data_norm, weights)
        classif = run_kmeans(X, n_clusters, M, N, seed=s)
        ce, cip = compute_CE_CIP(classif)
        ce_samples.append(ce)
        cip_samples.append(cip)

    mu_ce,  sigma_ce  = np.mean(ce_samples),  np.std(ce_samples)
    mu_cip, sigma_cip = np.mean(cip_samples), np.std(cip_samples)

    print(f"Calibration : mu_CE={mu_ce:.4f}, sigma_CE={sigma_ce:.4f} | "
          f"mu_CIP={mu_cip:.4f}, sigma_CIP={sigma_cip:.4f}")

    return mu_ce, sigma_ce, mu_cip, sigma_cip


def compute_score_normalized(ce, cip, mu_ce, sigma_ce, mu_cip, sigma_cip):
    """
    Score normalisé proposé par l'encadrant (équation 1) :
    score = (1/I) * sum_i (mu_i - z_i) / sigma_i
    Ici I=2, z_0=CE, z_1=CIP.
    """
    term_ce  = (mu_ce  - ce)  / sigma_ce  if sigma_ce  > 0 else 0
    term_cip = (mu_cip - cip) / sigma_cip if sigma_cip > 0 else 0
    return 0.5 * (term_ce + term_cip)


# RECUIT SIMULÉ AVEC DIRECTION PRIVILÉGIÉE (proposition 1)
def optimize_weights_directional(data_norm, gt, n_clusters=16, n_iter=500,
                                   seed=0, w_direction=0.5,
                                   calibration=None):
    """
    Recuit simulé à direction privilégiée

    Au lieu de tirer uniquement du bruit gaussien isotrope, on mélange :
      - une exploration dans la direction qui a fait progresser la solution
        la dernière fois (delta_x normalisé)
      - une exploration aléatoire pure (comme avant)

    w_direction : poids donné à la direction privilégiée (entre 0 et 1)

    calibration : tuple (mu_ce, sigma_ce, mu_cip, sigma_cip)
                  Si None, on utilise l'ancien score (1-CE)*(1-CIP)
    """
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    def score_fn(classif):
        ce, cip = compute_CE_CIP(classif)
        if calibration is not None:
            mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
            score = compute_score_normalized(ce, cip, mu_ce, sigma_ce,
                                              mu_cip, sigma_cip)
        else:
            score = (1 - ce) * (1 - cip)
        return score, ce, cip

    # Initialisation
    best_weights = np.ones(K)
    X0 = build_features(data_norm, best_weights)
    best_classif = run_kmeans(X0, n_clusters, M, N)
    best_score, best_ce, best_cip = score_fn(best_classif)
    best_oac = clustering_accuracy(gt, best_classif)

    # delta_x : dernière direction qui a amélioré la solution.
    # Initialisé à un vecteur aléatoire normalisé (pas de direction connue au début).
    delta_x = rng.randn(K)
    delta_x /= np.linalg.norm(delta_x)

    hist_ce, hist_cip, hist_oac, hist_score, hist_sigma = (
        [best_ce], [best_cip], [best_oac], [best_score], [])

    sigma, sigma_min = 0.5, 1e-4
    n_plateau = 0

    print(f"Départ | CE={best_ce:.4f} | CIP={best_cip:.4f} | "
          f"OAC={best_oac:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):

        # Tirage 1 : direction privilégiée (delta_x normalisé) * bruit scalaire
        b_prime = rng.randn()
        direction_term = b_prime * delta_x

        # Tirage 2 : bruit isotrope classique
        b_t = rng.randn(K)

        # Combinaison pondérée 
        noise = sigma * (w_direction * direction_term + (1 - w_direction) * b_t)

        new_weights = np.maximum(best_weights + noise, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        X_new = build_features(data_norm, new_weights)
        classif = run_kmeans(X_new, n_clusters, M, N, seed=t)
        score, ce, cip = score_fn(classif)
        oac = clustering_accuracy(gt, classif)

        hist_ce.append(ce)
        hist_cip.append(cip)
        hist_oac.append(oac)
        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            # Met à jour la direction privilégiée
            delta_x_new = new_weights - best_weights
            norm = np.linalg.norm(delta_x_new)
            if norm > 1e-4:
                delta_x = delta_x_new / norm

            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip, best_oac = classif, ce, cip, oac
            n_plateau = 0
            print(f"Iter {t:04d} | CE={ce:.4f} | CIP={cip:.4f} | "
                  f"OAC={oac:.4f} | score={score:.4f} | sigma={sigma:.4f} ✓")
        else:
            n_plateau += 1
            if n_plateau % 30 == 0:
                sigma = max(sigma * 0.5, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.1, 0

    print(f"\n=== Résultat final ===")
    print(f"OAC={best_oac:.4f} | CE={best_ce:.4f} | CIP={best_cip:.4f} | "
          f"score={best_score:.4f}")

    return {
        "classif": best_classif,
        "weights": best_weights,
        "hist_ce": hist_ce,
        "hist_cip": hist_cip,
        "hist_oac": hist_oac,
        "hist_score": hist_score,
        "hist_sigma": hist_sigma,
    }


# VISUALISATIONS
def plot_results(result, gt):
    classif = result["classif"]
    final_oac = result["hist_oac"][-1] if result["hist_oac"] else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(classif, cmap='jet')
    axes[0].set_title(f"Meilleure classification\nOAC={final_oac:.4f}")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # CE et CIP au cours des itérations 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(result["hist_cip"], color='darkorange')
    axes[0].set_title("Évolution de CIP")
    axes[0].set_xlabel("Itération")
    axes[0].grid(True)

    axes[1].plot(result["hist_ce"], color='steelblue')
    axes[1].set_title("Évolution de CE")
    axes[1].set_xlabel("Itération")
    axes[1].grid(True)

    axes[2].plot(result["hist_sigma"], color='purple')
    axes[2].set_title("Évolution de sigma")
    axes[2].set_xlabel("Itération")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Score vs OAC
    plt.figure(figsize=(7, 5))
    plt.scatter(result["hist_score"], result["hist_oac"],
               alpha=0.4, s=10, c='steelblue')
    plt.xlabel("Score")
    plt.ylabel("OAC")
    plt.title("Relation Score - OAC (direction privilégiée)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# POINT D'ENTRÉE
if __name__ == "__main__":
    data, gt = load_data('..\\dataset\\indianpinearray.npy',
                          '..\\dataset\\IPgt.npy')
    data_norm = normalize(data)
    n_clusters = 16

    # Étape 1 : calibration mu/sigma pour le score normalisé
    print("=== Calibration ===")
    calibration = calibrate_score(data_norm, n_clusters, n_samples=30)

    # Étape 2 : optimisation avec direction privilégiée + score normalisé
    print("\n=== Optimisation (direction privilégiée) ===")
    result = optimize_weights_directional(data_norm, gt, n_clusters=n_clusters, n_iter=300, w_direction=0.5, calibration=calibration)

    plot_results(result, gt)