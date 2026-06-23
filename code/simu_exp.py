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
  kvm.optimize_weights_recuit(kvm.data_norm, kvm.gt, n_clusters=n_clusters, n_iter=n_iter, seed=0)
  #save('exp2_cip.pkl',['best_classif', 'best_weights', 'history_ce', 'history_oac', 'history_score', 'history_sigma'],\
  #[best_classif, best_weights, history_ce, history_oac, history_score, history_sigma])
  #le programme compute_vraisemblance doit etre modifié pour changer CE et CIP
  save('exp2_ce.pkl',['best_classif', 'best_weights', 'history_ce', 'history_oac', 'history_score', 'history_sigma'],\
  [best_classif, best_weights, history_ce, history_oac, history_score, history_sigma])

def exp3(): 
  dc = load('exp2_cip.pkl')  
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


if __name__ == "__main__":  
  import warnings
  action()  
