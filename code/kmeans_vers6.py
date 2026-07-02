import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# ============================================================================
# CHARGEMENT ET NORMALISATION
# ============================================================================

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


# ============================================================================
# FEATURES (spectral + spatial, alpha contrôle l'importance du spatial)
# -- inchangé, utilisé par le pipeline kmeans / recuit simulé --
# ============================================================================

def build_features(data, weights=None, alpha=0.0):
    M, N, K = data.shape
    if weights is not None:
        data = data * weights[np.newaxis, np.newaxis, :]
    data_pad = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="reflect")

    coords_i = (np.arange(M) - M / 2) / (M / (2 * np.sqrt(3)))
    coords_j = (np.arange(N) - N / 2) / (N / (2 * np.sqrt(3)))

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
            if alpha > 0:
                spatial = np.array([alpha * coords_i[i-1],
                                     alpha * coords_j[j-1]])
                features.append(np.concatenate([spectral, spatial]))
            else:
                features.append(spectral)
    return np.array(features)


def run_kmeans(X, n_clusters, M, N, seed=42, init_method='custom', n_init=10):
    if init_method == 'custom':
        rng = np.random.RandomState(seed)
        init_idx = rng.choice(len(X), n_clusters, replace=False)
        init = X[init_idx]
        n_init = 1
    else:
        init = 'k-means++'

    kmeans = KMeans(n_clusters=n_clusters, init=init,
                     n_init=n_init, random_state=seed)
    return kmeans.fit_predict(X).reshape(M, N)


def reconstruct_image_from_centroids(data_norm, classif):
    M, N, K = data_norm.shape
    flat = data_norm.reshape(-1, K)
    labels = classif.flatten().astype(int)
    centroids = np.zeros((labels.max() + 1, K), dtype=np.float64)
    for c in range(centroids.shape[0]):
        mask = labels == c
        if np.any(mask):
            centroids[c] = flat[mask].mean(axis=0)
    recon = centroids[labels].reshape(M, N, K)
    return recon


def plot_reconstructed_image(recon, title="Reconstructed image"):
    M, N, K = recon.shape
    bands = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if K > 60 else [0, 1, min(2, K - 1)]
    img = np.stack([recon[:, :, bands[0]], recon[:, :, bands[1]], recon[:, :, bands[2]]], axis=2)
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    img = np.clip(img, vmin, vmax)
    img -= img.min(axis=(0, 1))
    img /= np.maximum(img.max(axis=(0, 1)), 1e-9)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"{title} (bands {bands})")
    plt.axis('off')
    plt.show()


# ============================================================================
# MÉTRIQUES GLOBALES (inchangées)
# ============================================================================

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


# ============================================================================
# CALIBRAGE / RECUIT SIMULÉ À DIRECTION PRIVILÉGIÉE (inchangé)
# ============================================================================

def calibrate_score(data_norm, n_clusters, alpha=0.0, n_samples=30, seed=42):
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)
    ce_samples, cip_samples = [], []
    for s in range(n_samples):
        weights = rng.uniform(0, 2, size=K)
        X = build_features(data_norm, weights, alpha=alpha)
        classif = run_kmeans(X, n_clusters, M, N, seed=42)
        ce, cip = compute_CE_CIP(classif)
        ce_samples.append(ce)
        cip_samples.append(cip)
    mu_ce,  sigma_ce  = np.mean(ce_samples),  np.std(ce_samples)
    mu_cip, sigma_cip = np.mean(cip_samples), np.std(cip_samples)
    return mu_ce, sigma_ce, mu_cip, sigma_cip


def compute_score_normalized(ce, cip, mu_ce, sigma_ce, mu_cip, sigma_cip):
    term_ce  = (mu_ce  - ce)  / sigma_ce  if sigma_ce  > 0 else 0
    term_cip = (mu_cip - cip) / sigma_cip if sigma_cip > 0 else 0
    return 0.5 * (term_ce + term_cip)


def optimize_weights_directional(data_norm, gt, n_clusters=16, n_iter=300, seed=0, w_direction=0.5,
                                  alpha=0.0, calibration=None, verbose=True):
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    def score_fn(classif):
        ce, cip = compute_CE_CIP(classif)
        oac = clustering_accuracy(gt, classif)
        if calibration is not None:
            mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
            normalized_score = compute_score_normalized(ce, cip, mu_ce, sigma_ce, mu_cip, sigma_cip)
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
        classif = run_kmeans(X_new, n_clusters, M, N, seed=t, init_method='k-means++', n_init=10)
        score, ce, cip, oac = score_fn(classif)

        hist_ce.append(ce); hist_cip.append(cip); hist_oac.append(oac)
        hist_score.append(score); hist_sigma.append(sigma)

        if score > best_score:
            delta_x_new = new_weights - best_weights
            norm = np.linalg.norm(delta_x_new)
            if norm > 1e-12:
                delta_x = delta_x_new / norm
            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip, best_oac = classif, ce, cip, oac
            n_plateau = 0
            if verbose:
                print(f"[alpha={alpha}] Iter {t:04d} | CE={ce:.4f} | CIP={cip:.4f} "
                      f"| OAC={oac:.4f} | score={score:.4f} \u2713")
        else:
            n_plateau += 1
            if n_plateau % 50 == 0:
                sigma = max(sigma * 0.9, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.1, 0

    if verbose:
        print(f"[alpha={alpha}] Final | OAC={best_oac:.4f} | CE={best_ce:.4f} "
              f"| CIP={best_cip:.4f} | score={best_score:.4f}\n")

    return {
        "alpha": alpha, "classif": best_classif, "weights": best_weights,
        "hist_ce": hist_ce, "hist_cip": hist_cip, "hist_oac": hist_oac,
        "hist_score": hist_score, "hist_sigma": hist_sigma,
        "final_oac": best_oac, "final_ce": best_ce,
    }


def sweep_alpha(data_norm, gt, alphas, n_clusters=16, n_iter=300,
                 w_direction=0.5, n_iter_calib=30):
    resultats = []
    for alpha in alphas:
        print(f"\n========== ALPHA = {alpha} ==========")
        calibration = calibrate_score(data_norm, n_clusters, alpha=alpha, n_samples=n_iter_calib)
        result = optimize_weights_directional(
            data_norm, gt, n_clusters=n_clusters, n_iter=n_iter,
            w_direction=w_direction, alpha=alpha, calibration=calibration, verbose=True)
        resultats.append(result)
    return resultats


def plot_best_result(best_result, gt, title_prefix="Meilleur résultat"):
    classif = best_result["classif"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(classif, cmap='jet')
    axes[0].set_title(f"{title_prefix} (alpha={best_result.get('alpha', '-')})\n"
                       f"OAC={best_result['final_oac']:.4f} | CE={best_result['final_ce']:.4f}")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# NOUVEAU (discussion_N260701, section 2) : CE / CIP RESTREINTS À UN
# SOUS-ENSEMBLE DE PIXELS M' (voisinage N(m) ∩ M')
# ============================================================================

def compute_CE_CIP_restricted(F, mask):
    """
    Version restreinte de CE et CIP (éq. 3-4 de discussion_N260701).

    F    : étiquetage (M,N) — binaire (candidat de seuillage) ou multi-classes
    mask : booléen (M,N) définissant le sous-ensemble M' considéré

    Le voisinage N(m) est restreint à N(m) ∩ M' : un pixel m' hors de M' n'est
    jamais pris en compte, ni comme voisin de référence, ni comme pixel évalué.
    """
    up, down    = np.roll(F, 1, axis=0), np.roll(F, -1, axis=0)
    left, right = np.roll(F, 1, axis=1), np.roll(F, -1, axis=1)
    mask_up, mask_down   = np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0)
    mask_left, mask_right = np.roll(mask, 1, axis=1), np.roll(mask, -1, axis=1)

    diff_up    = mask_up    & (F != up)
    diff_down  = mask_down  & (F != down)
    diff_left  = mask_left  & (F != left)
    diff_right = mask_right & (F != right)
    has_diff_neighbor = diff_up | diff_down | diff_left | diff_right

    same_up    = mask_up    & (F == up)
    same_down  = mask_down  & (F == down)
    same_left  = mask_left  & (F == left)
    same_right = mask_right & (F == right)
    has_same_neighbor = same_up | same_down | same_left | same_right
    has_neighbor = mask_up | mask_down | mask_left | mask_right

    # CE_M' : au moins un voisin (dans M') de classe différente
    ce_mask = mask & has_diff_neighbor
    # CIP_M' : le pixel a au moins un voisin dans M', et tous ces voisins diffèrent
    cip_mask = mask & has_neighbor & (~has_same_neighbor)

    denom = mask.sum()
    if denom == 0:
        return 0.0, 0.0
    return ce_mask.sum() / denom, cip_mask.sum() / denom


# ============================================================================
# NOUVEAU (discussion_N260701, section 3) : ARBRE DE DÉCISION PAR SEUILLAGE
# ============================================================================

def build_decision_tree(data, n_clusters, criterion='ce', weights=None,
                         max_thresholds=None, verbose=True):
    """
    Construit un arbre en scindant à chaque étape le noeud terminal contenant
    le plus de pixels, en choisissant la bande k et le seuil s qui minimisent
    CE_M' ou CIP_M' (calculés restreints au noeud courant).

    data           : image (M,N,K), typiquement data_norm
    n_clusters     : nombre de classes visées G (= nombre de noeuds terminaux)
    criterion      : 'ce', 'cip' ou 'ce_cip' (moyenne des deux)
    weights        : pondération optionnelle des bandes (sert à diversifier les
                      arbres d'une forêt, cf. build_forest)
    max_thresholds : si fourni, sous-échantillonne les seuils testés par bande
                      (accélère beaucoup le calcul, au prix d'une recherche
                      moins exhaustive)

    Retourne (classif, nodes, terminal_ids) :
        classif      : image (M,N) des étiquettes 0..G-1 (une par noeud terminal)
        nodes        : dict {id: noeud} avec mask, parent, split_band, split_thresh
        terminal_ids : liste des ids des noeuds terminaux (dans l'ordre de création)
    """
    M, N, K = data.shape
    data_w = data * weights[np.newaxis, np.newaxis, :] if weights is not None else data

    nodes = {0: {'id': 0, 'mask': np.ones((M, N), dtype=bool), 'parent': None}}
    terminal_ids = [0]
    frozen = set()   # noeuds qu'on ne peut plus scinder (bandes constantes, etc.)
    next_id = 1

    while len(terminal_ids) < n_clusters:
        candidates = [i for i in terminal_ids if i not in frozen]
        if not candidates:
            if verbose:
                print("Arrêt anticipé : plus aucun noeud ne peut être scindé "
                      f"({len(terminal_ids)} classes obtenues sur {n_clusters}).")
            break

        # étape 1 : noeud terminal contenant le plus de pixels
        n_id = max(candidates, key=lambda i: nodes[i]['mask'].sum())
        node = nodes[n_id]
        mask_n = node['mask']

        if mask_n.sum() <= 1:
            frozen.add(n_id)
            continue

        # étape 2 : recherche du (bande, seuil) minimisant le critère
        best_score, best_band, best_thresh = np.inf, None, None
        for k in range(K):
            band = data_w[:, :, k]
            values = np.unique(band[mask_n])
            if len(values) <= 1:
                continue
            candidate_values = values[:-1]  # le dernier seuil met tout d'un côté
            if max_thresholds is not None and len(candidate_values) > max_thresholds:
                idx = np.linspace(0, len(candidate_values) - 1, max_thresholds).astype(int)
                candidate_values = candidate_values[idx]

            for v in candidate_values:
                F = (band <= v).astype(np.int8)
                ce, cip = compute_CE_CIP_restricted(F, mask_n)
                score = ce if criterion == 'ce' else cip if criterion == 'cip' else 0.5 * (ce + cip)
                if score < best_score:
                    best_score, best_band, best_thresh = score, k, v

        if best_band is None:
            frozen.add(n_id)
            continue

        # étape 4 : scission du noeud
        band = data_w[:, :, best_band]
        mask_v = mask_n & (band <= best_thresh)   # condition vérifiée
        mask_f = mask_n & (band >  best_thresh)   # condition non vérifiée
        if mask_v.sum() == 0 or mask_f.sum() == 0:
            frozen.add(n_id)
            continue

        node['split_band']   = best_band
        node['split_thresh'] = float(best_thresh)
        node['score']        = best_score

        n_v = {'id': next_id, 'mask': mask_v, 'parent': n_id}; nodes[next_id] = n_v; next_id += 1
        n_f = {'id': next_id, 'mask': mask_f, 'parent': n_id}; nodes[next_id] = n_f; next_id += 1
        node['child_v'], node['child_f'] = n_v['id'], n_f['id']

        terminal_ids.remove(n_id)
        terminal_ids.extend([n_v['id'], n_f['id']])

        if verbose:
            print(f"Noeud {n_id:3d} scindé | bande={best_band:3d} | seuil={best_thresh:8.4f} "
                  f"| {criterion}={best_score:.4f} | tailles V={mask_v.sum():5d} F={mask_f.sum():5d} "
                  f"| noeuds terminaux={len(terminal_ids)}")

    classif = np.full((M, N), -1, dtype=int)
    for label, n_id in enumerate(terminal_ids):
        classif[nodes[n_id]['mask']] = label

    return classif, nodes, terminal_ids


# ============================================================================
# NOUVEAU (discussion_N260701, section 4) : FORÊT + CONSENSUS CLUSTERING
# ============================================================================

def build_forest(data, n_clusters, T=10, criterion='ce', seed=0,
                  weight_low=0.5, weight_high=1.5, max_thresholds=30, verbose=False):
    """
    Génère T arbres, chacun avec une pondération aléatoire différente des
    bandes spectrales (c'est ce qui rend les arbres différents les uns des
    autres). max_thresholds est fortement recommandé ici (T arbres complets
    sinon très coûteux).
    """
    rng = np.random.RandomState(seed)
    M, N, K = data.shape
    classifs = []
    for t in range(T):
        w = rng.uniform(weight_low, weight_high, size=K)
        print(f"--- Arbre {t + 1}/{T} ---")
        classif, _, _ = build_decision_tree(
            data, n_clusters, criterion=criterion, weights=w,
            max_thresholds=max_thresholds, verbose=verbose)
        classifs.append(classif)
    return classifs


def consensus_clustering(classifs, n_clusters, seed=0, max_iter=100, verbose=True):
    """
    Fusionne T segmentations en G régions en une segmentation consensus en G
    régions (Iterative Pairwise Consensus, discussion_N260701 section 4).

    Étape 1 : les "briques de base" sont les groupes de pixels ayant exactement
              le même vecteur d'étiquettes sur les T arbres.
    Étape 2 : D[h,h'] = fraction des arbres où les briques h et h' sont dans
              la même classe (co-association, notée "distance" dans le doc
              mais utilisée ici comme une similarité : plus elle est grande,
              plus h et h' doivent finir dans le même cluster consensus).
    Étape 3 : IPC — on affecte itérativement chaque brique au cluster avec
              lequel sa similarité moyenne est la plus grande, jusqu'à
              convergence.
    """
    T = len(classifs)
    M, N = classifs[0].shape
    stacked = np.stack(classifs, axis=-1).reshape(-1, T)

    # étape 1 : briques de base
    uniques, inverse = np.unique(stacked, axis=0, return_inverse=True)
    H = uniques.shape[0]
    if verbose:
        print(f"Consensus : {H} briques de base pour {M * N} pixels et {T} arbres.")

    # étape 2 : matrice de co-association H x H
    D = (uniques[:, None, :] == uniques[None, :, :]).mean(axis=2)

    # étape 3 : Iterative Pairwise Consensus
    rng = np.random.RandomState(seed)
    F = rng.randint(0, n_clusters, size=H)
    for it in range(max_iter):
        avg_sim = np.zeros((H, n_clusters))
        for g in range(n_clusters):
            members = (F == g)
            if members.sum() == 0:
                continue
            avg_sim[:, g] = D[:, members].mean(axis=1)
        new_F = np.argmax(avg_sim, axis=1)
        if np.array_equal(new_F, F):
            if verbose:
                print(f"Consensus convergé après {it} itérations.")
            break
        F = new_F

    brique_labels = F[inverse]
    classif_final = brique_labels.reshape(M, N)
    return classif_final


def plot_tree_vs_gt(classif, gt, oac, ce, title="Arbre de décision"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(classif, cmap='jet')
    axes[0].set_title(f"{title}\nOAC={oac:.4f} | CE={ce:.4f}")
    axes[0].axis('off')
    axes[1].imshow(gt, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    data, gt = load_data('..\\dataset\\indianpinearray.npy', '..\\dataset\\IPgt.npy')
    data_norm = normalize(data)
    n_clusters = 16

    # ------------------------------------------------------------------
    # 1) UN SEUL ARBRE (sans pondération, poids uniformes) — section 3
    # ------------------------------------------------------------------
    print("\n========== ARBRE UNIQUE (poids uniformes) ==========")
    classif_tree, nodes_tree, terminal_ids = build_decision_tree(
        data_norm, n_clusters, criterion='ce', max_thresholds=50, verbose=True)
    ce_tree, cip_tree = compute_CE_CIP(classif_tree)
    oac_tree = clustering_accuracy(gt, classif_tree)
    print(f"\nArbre unique | OAC={oac_tree:.4f} | CE={ce_tree:.4f} | CIP={cip_tree:.4f}")
    plot_tree_vs_gt(classif_tree, gt, oac_tree, ce_tree, title="Arbre unique")

    # ------------------------------------------------------------------
    # 2) FORÊT + CONSENSUS CLUSTERING — section 4
    #    (T modéré recommandé : la matrice de co-association grandit vite)
    # ------------------------------------------------------------------
    print("\n========== FORÊT (T=10 arbres) + CONSENSUS ==========")
    T = 10
    classifs = build_forest(data_norm, n_clusters, T=T, criterion='ce',
                             seed=0, max_thresholds=30, verbose=False)
    classif_consensus = consensus_clustering(classifs, n_clusters, seed=0, verbose=True)
    ce_cons, cip_cons = compute_CE_CIP(classif_consensus)
    oac_cons = clustering_accuracy(gt, classif_consensus)
    print(f"\nConsensus (T={T}) | OAC={oac_cons:.4f} | CE={ce_cons:.4f} | CIP={cip_cons:.4f}")
    plot_tree_vs_gt(classif_consensus, gt, oac_cons, ce_cons,
                     title=f"Forêt + consensus (T={T})")

    # ------------------------------------------------------------------
    # 3) (optionnel) Comparaison avec le pipeline kmeans / recuit simulé
    #    existant — décommenter pour relancer.
    # ------------------------------------------------------------------
    # alphas = [0.0, 1.5, 2, 3, 5.0]
    # resultats = sweep_alpha(data_norm, gt, alphas, n_clusters=n_clusters,
    #                          n_iter=200, w_direction=0.99, n_iter_calib=30)
    # best_result = max(resultats, key=lambda r: r["final_oac"])
    # plot_best_result(best_result, gt)

    print("\n===== RÉCAPITULATIF =====")
    print(f"Arbre unique     : OAC={oac_tree:.4f} | CE={ce_tree:.4f}")
    print(f"Forêt + consensus: OAC={oac_cons:.4f} | CE={ce_cons:.4f}")