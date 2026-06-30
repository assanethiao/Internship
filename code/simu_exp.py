import numpy as np
from typing import Any

def exp1(): 
  data = np.load('..\\dataset\\indianpinearray.npy')
  gt   = np.load('..\\dataset\\IPgt.npy')
  M, N, K = data.shape
  import kmeans_vers3_mod as kvm
  print(kvm.data.shape)

def exp2():
  n_clusters = 16
  n_iter     = 500
  print("=== Optimisation en cours ===")
  import kmeans_vers3_mod as kvm
  best_classif, best_weights, history_ce, history_oac, history_score, history_sigma =\
  kvm.optimize_weights_recuit(kvm.data_norm, kvm.gt, n_clusters=n_clusters, n_iter=n_iter, seed=42)
  #save('exp2_cip.pkl',['best_classif', 'best_weights', 'history_ce', 'history_oac', 'history_score', 'history_sigma'],\
  #[best_classif, best_weights, history_ce, history_oac, history_score, history_sigma])
  #le programme compute_vraisemblance doit etre modifié pour changer CE et CIP
  save('exp2_ce.pkl',['best_classif', 'best_weights', 'history_ce', 'history_oac', 'history_score', 'history_sigma'],\
  [best_classif, best_weights, history_ce, history_oac, history_score, history_sigma])

def exp3(): 
  dc = load('exp2_ce.pkl')  
  t = np.array(range(0,len(dc['history_score'])))
  #print(t)
  cip = 1-np.array(dc['history_score'])
  sigma = np.array(dc['history_sigma'])
  print(len(sigma),len(t))
  #print(cip)
  plt = debut()
  plt.close('all')
  fig,ax = plt.subplots()
  plt.tight_layout()
  ax.plot(t,cip)
  fig.show()
  fig.savefig('exp3_cip_fig1.png')
  fig,ax = plt.subplots()
  plt.tight_layout()
  ax.plot((t[1:]+t[:-1])/2,sigma)
  fig.show()
  fig.savefig('exp3_cip_fig2.png')


def exp4(): 
  dc = load('exp2_ce.pkl')  
  t = np.array(range(0,len(dc['history_score'])))
  ce = 1-np.array(dc['history_score'])
  sigma = np.array(dc['history_sigma'])
  plt = debut()
  plt.close('all')
  fig,ax = plt.subplots()
  plt.tight_layout()
  ax.plot(t,ce)
  fig.show()
  fig.savefig('exp4_ce_fig1.png')
  fig,ax = plt.subplots()
  plt.tight_layout()
  ax.plot((t[1:]+t[:-1])/2,sigma)
  fig.show()
  fig.savefig('exp4_ce_fig2.png')


def exp5(): 
  import kmeans_vers4 as kv
  data, gt = kv.load_data('../dataset/indianpinearray.npy',
                          '../dataset/IPgt.npy')
  data_norm = kv.normalize(data)
  n_clusters = 16
  M, N, K = data_norm.shape
  seed = 42
  rng = np.random.RandomState(seed)
  print("=== Calibration ===")
  calibration = kv.calibrate_score(data_norm, n_clusters, n_samples=30)
  weights = rng.randn(K)
  X_new = kv.build_features(data_norm, weights)
  classif = kv.run_kmeans(X_new, n_clusters, M, N, seed=seed)
  score, ce, cip = score_fn(classif,calibration=calibration)
  oac = kv.clustering_accuracy(gt, classif)
  print(f"CE={ce:.4f} | CIP={cip:.4f} | OAC={oac:.4f} | score={score:.4f} ")
  X_new1 = kv.build_features(data_norm, weights)
  classif1 = kv.run_kmeans(X_new1, n_clusters, M, N, seed=seed)
  score1, ce1, cip1 = score_fn(classif1,calibration=calibration)
  oac1 = kv.clustering_accuracy(gt, classif1)
  print(f"CE={ce1:.4f} | CIP={cip1:.4f} | OAC={oac1:.4f} | score={score1:.4f} ")
  X_new2 = kv.build_features(data_norm, weights)
  classif2 = kv.run_kmeans(X_new2, n_clusters, M, N, seed=seed)
  score2, ce2, cip2 = score_fn(classif2,calibration=calibration)
  oac2 = kv.clustering_accuracy(gt, classif2)
  print(f"CE={ce2:.4f} | CIP={cip2:.4f} | OAC={oac2:.4f} | score={score2:.4f} ")

def exp6(): 
  import kmeans_vers4 as kv
  def plot2(result,gt): 
    plt = debut()
    plt.close('all')
    fig,ax = plt.subplots()
    ax.plot(result["hist_cip"], color='darkorange')
    ax.set_xlabel("Iterations")
    plt.tight_layout()
    ax.grid(True)
    fig.show()
    fig.savefig('juin_29_fig4.png')
    print('figure : juin_29_fig4.png')

    fig,ax = plt.subplots()
    ax.plot(result["hist_ce"], color='steelblue')
    ax.set_xlabel("Iterations")
    plt.tight_layout()
    ax.grid(True)
    fig.show()
    fig.savefig('juin_29_fig5.png')
    print('figure : juin_29_fig5.png')

    fig,ax = plt.subplots()
    ax.plot(result["hist_sigma"], color='purple')
    ax.set_xlabel("Iterations")
    plt.tight_layout()
    ax.grid(True)
    fig.show()
    fig.savefig('juin_29_fig6.png')
    print('figure : juin_29_fig6.png')


  def optimize_weights_directional(data_norm, gt, n_clusters=16, n_iter=5000,
                                   seed=42, w_direction=0.5,
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
    import random as rd
    w_direction1 = 0.5
    M, N, K = data_norm.shape
    rng = np.random.RandomState(seed)

    def score_fn(classif):
        ce, cip = kv.compute_CE_CIP(classif)
        if calibration is not None:
            mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
            score = kv.compute_score_normalized(ce, cip, mu_ce, sigma_ce,
                                              mu_cip, sigma_cip)
        else:
            score = (1 - ce) * (1 - cip)
        return score, ce, cip

    # Initialisation
    best_weights = np.ones(K)
    X0 = kv.build_features(data_norm, best_weights)
    best_classif = kv.run_kmeans(X0, n_clusters, M, N)
    best_score, best_ce, best_cip = score_fn(best_classif)
    best_oac = kv.clustering_accuracy(gt, best_classif)

    # delta_x : dernière direction qui a amélioré la solution.
    # Initialisé à un vecteur aléatoire normalisé (pas de direction connue au début).
    delta_x = rng.randn(K)
    delta_x /= np.linalg.norm(delta_x)

    hist_ce, hist_cip, hist_oac, hist_score, hist_sigma, hist_is_best = (
        [best_ce], [best_cip], [best_oac], [best_score], [], [])

    sigma, sigma_min = 0.5, 5e-3
    n_plateau = 0

    print(f"Départ | CE={best_ce:.4f} | CIP={best_cip:.4f} | "
          f"OAC={best_oac:.4f} | score={best_score:.4f}")

    for t in range(1, n_iter + 1):

        # Tirage 1 : direction privilégiée (delta_x normalisé) * bruit scalaire
        b_prime = rng.randn()
        if n_plateau > 50: 
            delta_x = rng.randn(K)
            delta_x /= np.linalg.norm(delta_x)
        if rd.random() < 0.9: 
            direction_term = b_prime * delta_x
        else: 
            a=rng.randn(K)
            direction_term = b_prime * a/np.linalg.norm(a)

        # Tirage 2 : bruit isotrope classique
        b_t = rng.randn(K)

        # Combinaison pondérée 
        if rd.random() < 0.5: 
            w_direction2 = w_direction1
        else: 
            w_direction2 = 0.9
            
        noise = sigma * (w_direction2 * direction_term + (1 - w_direction2) * b_t)


        new_weights = np.maximum(best_weights + noise, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(K)

        X_new = kv.build_features(data_norm, new_weights)
        classif = kv.run_kmeans(X_new, n_clusters, M, N, seed=42)
        score, ce, cip = score_fn(classif)
        oac = kv.clustering_accuracy(gt, classif)

        hist_ce.append(ce)
        hist_cip.append(cip)
        hist_oac.append(oac)
        hist_score.append(score)
        hist_sigma.append(sigma)

        if score > best_score:
            # Met à jour la direction privilégiée
            w_direction1 = w_direction
            delta_x_new = new_weights - best_weights
            norm = np.linalg.norm(delta_x_new)
            if norm > 1e-4:
                delta_x = delta_x_new / norm

            best_score, best_weights = score, new_weights
            best_classif, best_ce, best_cip, best_oac = classif, ce, cip, oac
            n_plateau = 0
            print(f"Iter {t:04d} | CE={ce:.4f} | CIP={cip:.4f} | "
                  f"OAC={oac:.4f} | score={score:.4f} | sigma={sigma:.4f} ✓")
            is_best = True      
        else:
            is_best = False
            n_plateau += 1
            if n_plateau % 30 == 0:
                sigma = max(sigma * 0.5, sigma_min)
            if sigma <= sigma_min:
                sigma, n_plateau = 0.5, 0

        hist_is_best.append(is_best)        
        if np.mod(t,100)== 0: 
            result = {
                "classif": best_classif,
                "weights": best_weights,
                "hist_ce": hist_ce,
                "hist_cip": hist_cip,
                "hist_oac": hist_oac,
                "hist_score": hist_score,
                "hist_sigma": hist_sigma,
                "hist_is_best": hist_is_best,
            }        
            print(f"Iter {t:04d} | CE={ce:.4f} | CIP={cip:.4f} | "
                  f"OAC={oac:.4f} | score={score:.4f} | sigma={sigma:.4f} ✓")
            plot2(result,gt)      
            save('juin_29_result2.pkl',['result'],[result])
            

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
        "hist_is_best": hist_is_best,
    }

  data, gt = kv.load_data('../dataset/indianpinearray.npy',
                          '../dataset/IPgt.npy')
  data_norm = kv.normalize(data)
  n_clusters = 16

  # Étape 1 : calibration mu/sigma pour le score normalisé
  print("=== Calibration ===")
  calibration = kv.calibrate_score(data_norm, n_clusters, n_samples=30)

  # Étape 2 : optimisation avec direction privilégiée + score normalisé
  print("\n=== Optimisation (direction privilégiée) ===")
  result = optimize_weights_directional(data_norm, gt, n_clusters=n_clusters, n_iter=10**6, w_direction=0.999, calibration=calibration)
  plot2(result,gt)  


#######################################################################
#outils generaux

def debut()->Any:
  """import seb; plt,np,sig = seb.debut()"""  
  import matplotlib.pyplot as plt
  params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'legend.loc':'upper right'}
         
  plt.rcParams.update(params)
  return plt


def save(nom_fichier:str,list_nom_var:list[str],list_var:list[Any])->None:
  """sauvegarde sous format binaire la liste des variables indiquees dans list_var
  le fichier s'appelle nom_fichier
  les noms des variables doivent etre mises avec des apostrophes autour
  """
  import pickle
  assert len(nom_fichier)>4, 'nom_fichier doit faire plus que 4 lettres'
  assert nom_fichier[-4:] == '.pkl'
  assert type(list_var) == list
  assert type(list_nom_var[0]) == str
  assert len(list_var) == len(list_nom_var)
  list_var_=[list_nom_var,list_var]
  open_file = open(nom_fichier, "wb")
  pickle.dump(list_var_, open_file)
  open_file.close()
  
def load(nom_fichier:str)->dict[str,Any]:
  """lit le fichier binaire et renvoie un dictionnaire dont les clef
  sont les noms des variables enregistres. 
  """
  import pickle
  assert len(nom_fichier)>4, 'nom_fichier doit faire plus que 4 lettres'
  assert nom_fichier[-4:] == '.pkl'
  file = open(nom_fichier, "rb")
  
  list_var = pickle.load(file)
  dc={}
  for k in range(len(list_var[0])):
    dc[list_var[0][k]]=list_var[1][k]
  file.close()
  return dc
  

def sys_read():
  """lit le premier argument rentres en ligne de commande retourne True 
     et cet argument ou False et rien s'il n'y rien
  """
  import sys
  if not len(sys.argv) in [2,3]:
    return False,""
  elif len(sys.argv)==2:
    return True,sys.argv[1],1
  else: 
    coeur = int(sys.argv[2])
    return True,sys.argv[1],coeur


def action():
  global coeur
  ok,arg,coeur = sys_read()
  if ok:
    if arg in globals():    
      globals()[arg]()
    else: 
      print("il faut que la fonction soit définie")
  else: print("il faut qu'il y ait le nom de la fonction")



def score_fn(classif,calibration=[]):
  import kmeans_vers4 as kv
  ce, cip = kv.compute_CE_CIP(classif)
  if calibration is not None:
    mu_ce, sigma_ce, mu_cip, sigma_cip = calibration
    score = kv.compute_score_normalized(ce, cip, mu_ce, sigma_ce,
                                      mu_cip, sigma_cip)
  else:
    score = (1 - ce) * (1 - cip)
  return score, ce, cip


if __name__ == "__main__":  
  import warnings
  action()  