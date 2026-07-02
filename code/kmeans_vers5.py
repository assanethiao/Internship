"""
kmeans_vers5.py
Implémente les propositions du document du 30 juin 2026 :
  - Métriques CE/CIP normalisées par sqrt(taille) (équations 1 et 2)
  - Architecture en échelle (ladder) : un weights différent par classe
  - Recuit simulé à direction privilégiée 
  - K-means déterministe (seed fixe, n_init=1)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


# 1. CHARGEMENT ET NORMALISATION
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


# 2. FEATURES (spectral uniquement, sans alpha)
#    Le document du 30 juin explique pourquoi alpha est problématique :
#    il pousse les classes à être des disques, ce qui ne correspond
#    pas au ground truth qui a des classes multi-régions.
def build_features(data, weights=None):
    """Concatène [centre, haut, bas, gauche, droite] pour chaque pixel."""
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
    """
    K-means strictement déterministe.
    seed fixe + n_init=1 garantissent le même résultat
    pour les mêmes weights. Requis par le recuit simulé
    (doc du 30 juin).
    """
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(len(X), n_clusters, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, init=X[init_idx],
                    n_init=1, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)


# 3. MÉTRIQUES NORMALISÉES PAR SQRT(TAILLE)
#    Nouvelles équations (1) et (2) du document du 30 juin.
#    Corrige le biais vers les classes très déséquilibrées.
def compute_CE_CIP_normalise(classification):
    """
    Calcule CE et CIP normalisés par sqrt(taille de classe),
    comme proposé dans les équations (1) et (2) du doc du 30 juin.

    Pour chaque classe A :
      CE_norm(A)  = min( contour_dans_A / sqrt(|A|),
                         contour_hors_A / sqrt(|M\\A|) )
      CIP_norm(A) = min( isoles_dans_A  / sqrt(|A|),
                         isoles_hors_A  / sqrt(|M\\A|) )

    """
    M, N = classification.shape
    total = M * N

    # Calcul des bords (4-connexe) — identique à avant
    up    = np.roll(classification,  1, axis=0)
    down  = np.roll(classification, -1, axis=0)
    left  = np.roll(classification,  1, axis=1)
    right = np.roll(classification, -1, axis=1)
    bord = ((classification != up)   | (classification != down) |
            (classification != left) | (classification != right))

    # Calcul des pixels isolés (8-connexe) — identique à avant
    pad = np.pad(classification, 1, mode='edge')
    isolated = np.ones((M, N), dtype=bool)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor = pad[1+di:M+1+di, 1+dj:N+1+dj]
            isolated &= (classification != neighbor)

    classes = np.unique(classification)
    ce_norm_vals  = []
    cip_norm_vals = []

    for c in classes:
        masque_A = (classification == c)
        taille_A = masque_A.sum()
        taille_B = total - taille_A  # complément de A

        if taille_A == 0 or taille_B == 0:
            continue

        # Contours dans A et hors A
        contour_dans_A = (bord & masque_A).sum()
        contour_hors_A = (bord & ~masque_A).sum()

        ce_A = contour_dans_A / np.sqrt(taille_A)
        ce_B = contour_hors_A / np.sqrt(taille_B)
        ce_norm_vals.append(max(ce_A, ce_B))

        # Pixels isolés dans A et hors A
        isoles_dans_A = (isolated & masque_A).sum()
        isoles_hors_A = (isolated & ~masque_A).sum()

        cip_A = isoles_dans_A / np.sqrt(taille_A)
        cip_B = isoles_hors_A / np.sqrt(taille_B)
        cip_norm_vals.append(max(cip_A, cip_B))

    ce_norm  = np.mean(ce_norm_vals)  if ce_norm_vals  else 0
    cip_norm = np.mean(cip_norm_vals) if cip_norm_vals else 0

    return ce_norm, cip_norm


def clustering_accuracy(gt, pred):
    gt_f, pred_f = gt.flatten(), pred.flatten()
    mask = gt_f != 0
    cm = confusion_matrix(gt_f[mask], pred_f[mask])
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()


# 4. CALIBRAGE DU SCORE (tirages aléatoires purs)
#    Estime mu et sigma pour normaliser le score
#    (équation 1 du doc du 19 juin).
def calibrate_score(data_norm, n_clusters, n_samples=30, seed=42):
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)
    ce_samples, cip_samples = [], []
    for s in range(n_samples):
        weights = rng.uniform(0, 2, size=K)
        X = build_features(data_norm, weights)
        classif = run_kmeans(X, n_clusters, M, N, seed=42)
        ce, cip = compute_CE_CIP_normalise(classif)
        ce_samples.append(ce)
        cip_samples.append(cip)
    mu_ce,  sigma_ce  = np.mean(ce_samples),  np.std(ce_samples)
    mu_cip, sigma_cip = np.mean(cip_samples), np.std(cip_samples)
    print(f"Calibration : mu_CE={mu_ce:.4f} sigma_CE={sigma_ce:.4f} | "
          f"mu_CIP={mu_cip:.4f} sigma_CIP={sigma_cip:.4f}")
    return mu_ce, sigma_ce, mu_cip, sigma_cip


def compute_score(ce, cip, calibration):
    """
    Score normalisé (équation 1 du doc du 19 juin)
    """
    mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
    term_ce  = (mu_ce  - ce)  / sigma_ce  if sigma_ce  > 0 else 0
    term_cip = (mu_cip - cip) / sigma_cip if sigma_cip > 0 else 0
    return 0.5 * (term_ce + term_cip)


# 5. RECUIT SIMULÉ À DIRECTION PRIVILÉGIÉE
#    Utilisé à chaque étape de l'architecture en échelle.
#    Score = CE_norm + CIP_norm normalisés
def optimize_one_step(data_norm, mask_pixels, calibration,
                       n_clusters=2, n_iter=150, seed=42,
                       w_direction=0.5, verbose=True, label=""):
    """
    Optimise un vecteur weights pour séparer les pixels actifs
    (mask_pixels=True) en n_clusters groupes.

    Utilisé dans l'architecture en échelle : à chaque étape,
    on n'optimise que sur les pixels restants (mask_pixels).

    Retourne le meilleur classif trouvé et les weights associés.
    """
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    best_weights = np.ones(K)
    X_full = build_features(data_norm, best_weights)

    # K-means appliqué uniquement sur les pixels actifs
    idx_actifs = np.where(mask_pixels.flatten())[0]

    def kmeans_sur_actifs(weights, seed_km):
        X = build_features(data_norm, weights)
        X_actifs = X[idx_actifs]
        km = KMeans(n_clusters=n_clusters, init='random',
                    n_init=1, random_state=seed_km)
        labels_actifs = km.fit_predict(X_actifs)
        full_labels = np.full(M * N, -1, dtype=int)
        full_labels[idx_actifs] = labels_actifs
        return full_labels.reshape(M, N)

    best_classif = kmeans_sur_actifs(best_weights, seed)
    # On évalue uniquement sur les pixels actifs
    classif_actifs = best_classif.copy()
    classif_actifs[~mask_pixels] = 0
    ce, cip = compute_CE_CIP_normalise(classif_actifs)
    best_score = compute_score(ce, cip, calibration)
    best_ce, best_cip = ce, cip

    delta_x = rng.randn(K)
    delta_x /= np.linalg.norm(delta_x)

    hist_score, hist_sigma = [best_score], []
    sigma, sigma_min = 0.5, 0.05
    n_plateau = 0

    if verbose:
        print(f"  [{label}] Départ | CE={ce:.4f} | CIP={cip:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):
        b_prime = rng.randn()
        b_t = rng.randn(K)
        noise = sigma * (w_direction * b_prime * delta_x
                          + (1 - w_direction) * b_t)
        new_weights = np.maximum(best_weights + noise, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        classif = kmeans_sur_actifs(new_weights, seed)
        classif_eval = classif.copy()
        classif_eval[~mask_pixels] = 0
        ce, cip = compute_CE_CIP_normalise(classif_eval)
        score = compute_score(ce, cip, calibration)

        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            delta_x_new = new_weights - best_weights
            norme = np.linalg.norm(delta_x_new)
            if norme > 1e-12:
                delta_x = delta_x_new / norme
            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip = classif, ce, cip
            n_plateau = 0
            if verbose:
                print(f"  [{label}] Iter {t:03d} | CE={ce:.4f} | "
                      f"CIP={cip:.4f} | score={score:.4f} ✓")
        else:
            n_plateau += 1
            if n_plateau % 50 == 0:
                sigma = max(sigma * 0.5, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.5, 0

    if verbose:
        print(f"  [{label}] Final | CE={best_ce:.4f} | CIP={best_cip:.4f} | "
              f"score={best_score:.4f}")

    return best_classif, best_weights, hist_score, hist_sigma


# 6. ARCHITECTURE EN ÉCHELLE (proposition 1 du doc du 30 juin)
#
#    À chaque étape c (de 1 à n_classes-1) :
#      - K-means à 2 clusters sur les pixels restants
#        avec un vecteur weights_c optimisé indépendamment
#      - Le cluster le plus petit = classe c (extrait)
#      - Les pixels restants passent à l'étape suivante
#    Dernière étape : les pixels restants = classe n_classes
#
#    Avantage : chaque weights_c n'a qu'UN objectif à la fois
#    (séparer une classe du reste), sans contradictions entre classes.
def ladder_clustering(data_norm, gt, calibration,
                       n_classes=16, n_iter_per_step=200,
                       w_direction=0.99):
    """
    Architecture en échelle comme proposé dans le document du 30 juin,
    section 5, première proposition.

    À chaque étape, un vecteur weights différent est optimisé
    pour extraire une classe parmi les pixels restants.
    """
    M, N, K = data_norm.shape
    F = np.zeros((M, N), dtype=int)
    mask_restant = np.ones((M, N), dtype=bool)

    all_weights = []
    all_hist_score = []
    oac_steps = []

    for c in range(1, n_classes):
        n_restant = mask_restant.sum()
        print(f"\n{'='*50}")
        print(f"Étape {c}/{n_classes-1} | Pixels restants : {n_restant} "
              f"({100*n_restant/(M*N):.1f}%)")

        # Optimisation : trouver les weights qui séparent bien
        # une classe des pixels restants
        best_classif, weights_c, hist_score, _ = optimize_one_step(
            data_norm, mask_restant, calibration,
            n_clusters=2,
            n_iter=n_iter_per_step,
            seed=42,
            w_direction=w_direction,
            label=f"classe {c}"
        )

        # Parmi les 2 clusters, on extrait le PLUS PETIT
        # (en accord avec la remarque du doc : la classe extraite
        # doit être plus petite que la classe restante)
        labels_actifs = best_classif[mask_restant]
        counts = np.bincount(labels_actifs[labels_actifs >= 0])
        if len(counts) < 2:
            # Cas dégénéré : un seul cluster trouvé
            print(f"  Attention : un seul cluster à l'étape {c}, on passe.")
            continue

        minority_label = np.argmin(counts)

        # Masque des pixels extraits = pixels actifs du cluster minoritaire
        mask_extrait = mask_restant & (best_classif == minority_label)

        F[mask_extrait] = c
        mask_restant = mask_restant & (~mask_extrait)

        all_weights.append(weights_c)
        all_hist_score.append(hist_score)

        oac_c = clustering_accuracy(gt, F)
        oac_steps.append(oac_c)
        print(f"  Classe {c} extraite : {mask_extrait.sum()} pixels | "
              f"OAC partiel = {oac_c:.4f}")

    # Les pixels restants forment la dernière classe
    F[mask_restant] = n_classes
    print(f"\nClasse {n_classes} (reste) : {mask_restant.sum()} pixels")

    oac_final = clustering_accuracy(gt, F)
    print(f"\n=== OAC final (architecture échelle) = {oac_final:.4f} ===")

    return F, all_weights, all_hist_score, oac_steps


# 7. VISUALISATIONS
def plot_results(classif, gt, oac_steps, all_hist_score):
    oac_final = clustering_accuracy(gt, classif)
    ce_final, cip_final = compute_CE_CIP_normalise(classif)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(classif, cmap='jet')
    axes[0].set_title(f"Architecture échelle\nOAC={oac_final:.4f} | "
                      f"CE={ce_final:.4f} | CIP={cip_final:.4f}")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # OAC partiel à chaque étape de l'échelle
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(oac_steps)+1), oac_steps, 'o-', color='steelblue')
    plt.xlabel("Étape (classe extraite)")
    plt.ylabel("OAC partiel")
    plt.title("Évolution de l'OAC à chaque étape de l'échelle")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Score de vraisemblance à chaque étape
    fig, axes = plt.subplots(1, min(4, len(all_hist_score)),
                              figsize=(4 * min(4, len(all_hist_score)), 4))
    if len(all_hist_score) == 1:
        axes = [axes]
    for i, hist in enumerate(all_hist_score[:4]):
        axes[i].plot(hist, color='darkorange')
        axes[i].set_title(f"Score étape {i+1}")
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()


# 8. POINT D'ENTRÉE
if __name__ == "__main__":
    data, gt = load_data('..\\dataset\\indianpinearray.npy',
                          '..\\dataset\\IPgt.npy')
    data_norm = normalize(data)
    M, N, K = data_norm.shape
    n_classes = 16

    # Calibration sur la totalité de l'image
    print("=== Calibration ===")
    calibration = calibrate_score(data_norm, n_clusters=2,
                                   n_samples=20, seed=42)

    # Architecture en échelle
    print("\n=== Architecture en échelle ===")
    classif, all_weights, all_hist_score, oac_steps = ladder_clustering(
        data_norm, gt, calibration,
        n_classes=n_classes,
        n_iter_per_step=200,
        w_direction=0.99
    )

    plot_results(classif, gt, oac_steps, all_hist_score)