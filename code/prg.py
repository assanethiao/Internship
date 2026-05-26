###############################################################
#sys pour demarrage

def sys__actualiser_debut():
  """copie les lignes de codes qui suivent dans le programme 
  sys__debut.py du repertoire c:/users/invite
  """
  lignes="""
def sys__debut():
  import sys
  sys.path.append('C:/A/SIMU/SIMU_Z/FENG/prg')
  sys.path.append('c:/users/invite')
  import os
  os.chdir('C:/A/SIMU/SIMU_Z/FENG/')
  cmd=r'import prg; plt,np=prg.debut(); import importlib; import ess; import os'
  return cmd
  
def sd(): 
  return sys__debut()  
  
if __name__ == '__main__':
  import sys__feng
  sys__feng.sd()
  print('exec(sys__feng.sd())')
  """
  print(lignes)
  with open('c:/users/invite/sys__feng.py', 'wt') as f:
    f.write(lignes)


##########################################################
#fonctions specifiques
def debut():
  """import seb; plt,np,sig = seb.debut()"""  
  import matplotlib.pyplot as plt
  params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'legend.loc':'upper right'}
         
  plt.rcParams.update(params)
  #plt.rcParams['text.usetex'] = True
  import numpy as np
  #import scipy.signal as sig
  return plt,np

def pine():
  """retrieves the hyperspectral image dataset and the ground truth"""
  import numpy as np # linear algebra
  #import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    #pip install pandas
  #import tifffile as tiff
    #pip install tifffile
  #img = tiff.imread('./donnees/bundle/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_EW-line_R.tif')
  #img = tiff.imread('./donnees/bundle/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif')
  img = np.load('./donnees/indianpinearray.npy')
  # print(img.shape)
  gdt = np.load('./donnees/IPgt.npy')
  # print(gdt.shape)
  return img,gdt

########################################################
#entropy
def en__g(x):
  """solution of y*log(y)+y=x 
  for x in [0, +infty)
  """
  from scipy import stats, optimize
  import numpy as np
  objective_function = lambda y: y*np.log(y)+y-x
  solution = optimize.newton(objective_function, x0=0.68)
  return solution

def en__g2(p,epsilon):
  """qi=exp(-lambda pi)/sum(exp(-lambda pi)) tel que epsilon = sum(pi qi)
  renvoie la valeur de lambda
  """
  from scipy import stats, optimize
  import numpy as np
  assert epsilon>np.min(p)
  objective_function = lambda x: np.sum(p*np.exp(-np.exp(x)*p))-epsilon*np.sum(np.exp(-np.exp(x)*p))
  print(objective_function(-10**5),objective_function(10),objective_function(10**5))
  assert objective_function(10**5)>0
  solution = np.log(optimize.toms748(objective_function, 1, 10**5))
  # solution = optimize.newton(objective_function, x0=0.1)
  return solution

def en__g3(p,epsilon):
  """qi=sum(exp(-lambda pi)) tel que epsilon = sum(pi qi)
  renvoie la valeur de lambda
  """
  """utilisation de cvxpy"""
  import cvxpy as cp
  import numpy as np
  N=len(p)
  q = cp.Variable(N)
  a = np.ones((1,N))
  objective = cp.Maximize(cp.sum(cp.entr(q)))
  constraints = [cp.sum(q) == 1,q>=0,cp.sum(cp.multiply(p,q))<=epsilon]
  problem=cp.Problem(objective,constraints)
  try: 
    problem.solve()
    if problem.status == 'optimal':
      epsilon_val = np.sum(p*q.value)
      return True,q.value,epsilon_val
    else:
      return False,np.nan,np.nan  
  except:
    return False,np.nan,np.nan   

def en__g4(p):
  """
  maximize {/sum_n / frac{q_n}{q_n+p_n}} subjected to q_n/geq 0 and {/sum_n q_n}=1
  """
  import numpy as np
  ind_sort = np.argsort(p)
  p1 = p[ind_sort]
  N  = len(p)
  q1 = np.zeros(N)
  j_opt,P_opt,q_opt=np.nan,-np.inf,np.nan
  for n in range(1,N+1): 
    an = np.sum(np.sqrt(p1[:n]))
    bn = np.sum(p1[:n])
    mu = an/(1+bn)
    q1[:n] = np.copy(np.sqrt(p1[:n])/mu - p1[:n])
    q1[n:] = 0
    if any(q1<0): 
      continue
    P = n - an**2/(1+bn)
    if P_opt < P : 
      P_opt,j_opt = P,n
      q_opt = np.copy(q1)
  q=np.zeros(N) 
  q[ind_sort]=np.copy(q_opt)
  return P_opt,j_opt,q

##########################################################
#manipulation image
def im__shuffle(img1):
  """change aleatoirement l'ordre des pixels
  def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]
  """
  import numpy as np
  assert len(img1.shape) == 3, img1.shape
  # access = np.arange(0,img1.shape[0]*img1.shape[1])
  # np.random.shuffle(access)
  # img2 = np.zeros(img1.shape)
  # for i in range(len(access)):
    # img2[i
  def access_def(img_shape):
    import numpy as np
    n=0
    access = np.zeros((img_shape[0],img_shape[1]),dtype=object)
    for i in range(img_shape[0]): 
      for j in range(img_shape[1]): 
        access[i,j] = (i,j)
    return access

  def access_wrt(img1,access):
    import numpy as np
    img2 = np.zeros(img1.shape)
    for i in range(img1.shape[0]): 
      for j in range(img1.shape[1]): 
        img2[access[i,j][0],access[i,j][1],:] = img1[i,j,:]
    return img2

  def access_test():
    access = access_def(img1.shape)
    assert isinstance(access[0][0][0],(int,np.integer,np.uint)), (type(access[0][0][0]))
    img2 = access_wrt(img1,access)
    assert (img1 == img2).any()
  
  access_test()
  access = access_def(img1.shape)
  np.random.shuffle(access)  
  img2 = access_wrt(img1,access)
  return img2

####################################################################
def fct(Y_p,Y_l,X_p,X_l):
  """creates two functions one with the minimum entropy conditionally to each bin 
  and the average conditionally to each bin"""
  import numpy as np
  M=len(Y_l)-1
  assert len(Y_p)==len(X_p)
  assert len(Y_l)-1==len(X_l)-1
  X2Y_min,X2Y_moy = np.zeros(M), np.zeros(M)
  for m in range(M): 
    ind = np.where((X_l[m]<=X_p)&(X_p<X_l[m+1]))
    if 0==len(Y_p[ind]):
      X2Y_min[m] = np.nan
      X2Y_moy[m] = np.nan
    else:
      X2Y_min[m] = min(Y_p[ind])
      X2Y_moy[m] = np.mean(Y_p[ind])
  return X2Y_min,X2Y_moy
  
def zone_fun(A,B,c,M):
  """checks where M is
  if it is in  the rectangle ABx(-c)  returns -1
  else if in ABx(c) returns 1
  else return 0
  """
  import prg
  plt,np = prg.debut()
  assert A.shape == (2,), A
  assert B.shape == (2,), B
  assert np.isscalar(c)    , c
  assert M.shape[0] == 2, M
  x = np.sum((M-A)*(B-A))
  y = np.linalg.det([M-A, B-A])
  z = np.sum((B-A)*(B-A))
  if not x>=0:
    return 0
  elif not x <= z:
    return 0
  elif 0<=y<=c*np.sqrt(z):
    return 1
  elif -c*np.sqrt(z)<=y<=0:
    return -1
  else:
    return 0

def ml_predict(clf,img):
  import numpy as np
  M,N = img.shape[:-1]
  img2 = np.zeros((M,N))
  for m in range(M): 
    for n in range(N):
      img2[m,n] = clf.predict(np.array([img[m,n,:]]))
  return img2

def ml_test1(clf,X):
  import numpy as np
  L=X.shape[0]
  Y=np.zeros(L)
  for l in range(L):
    Y[l]=clf.predict(np.array([X[l,:]]))
  return Y

def ml_train1(X_tr,Y_tr):
  from sklearn import svm
  assert X_tr.shape[0]==len(Y_tr)
  #pip install -U scikit-learn
  clf = svm.SVC()
  clf.fit(X_tr,Y_tr)  #kernel=RBF C=1 gamma=scale
  # print(clf._gamma)
  return clf

def ml_train_sel2(X_tr,Y_tr):
  from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
  from sklearn.svm import SVC
  import numpy as np
  gamma_range = np.logspace(-9, 1, 13)
  param_grid = dict(gamma=gamma_range)
  cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
  grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
  grid.fit(X_tr, Y_tr)
  return grid.best_params_['gamma']

def ml_train2(X_tr,Y_tr,gamma_val):
  from sklearn import svm
  assert X_tr.shape[0]==len(Y_tr)
  #pip install -U scikit-learn
  clf = svm.SVC(gamma=gamma_val)
  clf.fit(X_tr,Y_tr)  #kernel=RBF C=1 gamma=scale
  assert gamma_val == clf._gamma
  return clf

def ml_dataset(img,img2):
  import numpy as np
  assert len(img.shape)==3
  assert len(img2.shape)==2
  assert img.shape[:2]==img2.shape
  K=img.shape[2]
  ind1,ind2=np.where(np.abs(img2)==1)
  L=len(ind1)
  X=np.zeros((L,K))
  Y=np.zeros(L)
  for l in range(L):
    X[l,:]=img[ind1[l],ind2[l],:]
    Y[l]=(img2[ind1[l],ind2[l]]==1)
  assert type(X)==np.ndarray
  assert X.shape[0]==len(ind1)
  assert X.shape[1]==K
  return X,Y

def zone_fun2(I,rho,theta,M):
  """checks where M is
  if it is in  the rectangle ABx(-rho) and rho distance from I  returns -1
  else if in ABx(rho) and rho distance from I returns 1
  else return 0
  A is rho ei(pi/2+theta) and B is rho ei(theta-pi/2)
  """
  assert False
  import prg
  plt,np = prg.debut()
  assert I.shape == (2,), I
  assert np.isscalar(rho)    , rho
  assert np.isscalar(theta)    , theta
  assert M.shape[0] == 2, M
  A=np.array([-rho*np.sin(theta),rho*np.cos(theta)])+I
  B=np.array([rho*np.sin(theta),-rho*np.cos(theta)])+I
  d=np.linalg.vector_norm(M-I)
  c=d
  x = np.sum((M-A)*(B-A))
  y = np.linalg.det([M-A, B-A])
  z = np.sum((B-A)*(B-A))
  if not x>=0:
    return 0
  elif not x <= z:
    return 0
  elif (0<=y<=c*np.sqrt(z)) and (d<=rho):
    return 1
  elif (-c*np.sqrt(z)<=y<=0) and (d<=rho):
    return -1
  else:
    return 0

def zone_fun3(I,rho,theta,M):
  """checks where M is
  if close to pi/2 -pi/2 at h distance of the line returns 0
  if it is in demi-cercle -pi/2 pi/2 et dans rho returns 1
  if it is in demi-cercle +pi/2 pi et -pi -pi/2 et dans rho returns -1
  I center of cercle
  theta angle with respect to horizontal line
  """
  import prg
  plt,np = prg.debut()
  assert I.shape == (2,), I
  assert np.isscalar(rho)    , rho
  assert np.isscalar(theta)    , theta
  assert M.shape[0] == 2, M
  d=geo_norm(M-I)
  theta_M=geo_angle(M-I)
  k=np.floor((theta_M-theta+np.pi)/np.pi/2)
  theta_=theta_M-theta-k*2*np.pi
  assert -np.pi<=theta_<=np.pi, theta_
  if d>=rho:
    return 0
  elif d*np.abs(np.cos(theta_))<2:
    return 0
  elif np.abs(theta_)<=np.pi/2:
    return 1
  else:
    return -1


def zone_img(img_shape,A,B,c):
  """creates a -1,0,1 image using zone_fun and img_shape
  """
  import prg
  plt,np = prg.debut()
  img2 = np.zeros(img_shape)
  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      P=np.array([i,j])
      img2[i][j] = prg.zone_fun(A,B,c,P)
  return img2

def zone_img_fun(img_shape,fun):
  """creates a -1,0,1 image using zone_fun and img_shape
  """
  import prg
  plt,np = prg.debut()
  img2 = np.zeros(img_shape)
  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      P=np.array([i,j])
      img2[i][j] = fun(P)
  return img2

  
def zone_ds_img(img_shape,A,B,c):  
  """checks if zone_img is in img
  """
  import numpy as np
  assert len(img_shape) == 2, img_shape
  #print(img_shape),
  M,N = img_shape
  test = lambda P: (0 <= P[0] <= M) and (0 <= P[1] <= N)
  r90 = lambda P: np.array([P[1],-P[0]])
  # print(test(A),test(B),test(A+r90(B-A)),test(A-r90(B-A)),test(B+r90(A-B)),test(B-r90(A-B)))
  epsi=np.linalg.vector_norm(B-A)
  u=c*r90(B-A)/epsi
  assert np.abs(np.linalg.vector_norm(u)-c)<1e-8, (u,np.linalg.vector_norm(u),c)
  return test(A) and test(B) and test(A+u) and test(A-u) and test(B+u) and test(B-u)

def zone_ds_img2(img_shape,I,rho,theta):  
  """checks if zone_img is in img
  """
  import numpy as np
  assert len(img_shape) == 2, img_shape
  M,N = img_shape
  test = lambda P,rho: (0+rho <= P[0] < M-rho) and (0+rho <= P[1] < N-rho)
  return test(I,rho)


def zone_choix(img_shape): 
  import numpy as np
  assert len(img_shape)==2
  M,N=img_shape
  assert M==N
  K=10**3
  for k in range(K):
    A=np.random.uniform(0,M,2)
    B=np.random.uniform(0,M,2)
    c=np.random.uniform(2,M)  
    if zone_ds_img(img_shape,A,B,c):
      if not np.sum((zone_img((M,M),A,B,c)==1)&(zone_img((M,M),A,B,1)==0))>=1: 
        continue
      else:
        return A,B,c
  assert False, (A,B,c,img_shape)

def zone_choix2(img_shape): 
  import numpy as np
  assert len(img_shape)==2
  M,N=img_shape
  assert M==N
  K=10**3
  for k in range(K):
    I=np.random.uniform(0,M,2)
    rho=np.random.uniform(0,M/2)
    theta=np.random.uniform(-np.pi,np.pi)  
    if rho<=15:
      continue
    if zone_ds_img2((M,M),I,rho,theta):
      return I,rho,theta
  assert False, (I,rho,theta,img_shape)

def zone_choix3(img_shape,fun_ok): 
  import numpy as np
  assert len(img_shape)==2
  M,N=img_shape
  assert M==N
  K=10**3
  for k in range(K):
    I=np.random.uniform(0,M,2)
    rho=np.random.uniform(0,M/2)
    theta=np.random.uniform(-np.pi,np.pi)  
    if not fun_ok(I,rho,theta):
      continue
    if zone_ds_img2((M,M),I,rho,theta):
      return I,rho,theta
  assert False, (I,rho,theta,img_shape)

  
def zone_find(img,K):
  import numpy as np
  import prg
  spec_moy,spec_std = img_spec(img)
  img1 = img_normalize(img,spec_moy,spec_std)
  if prg.isfile('data_exp5.pkl'):
    dc=prg.load('data_exp5.pkl')
    a_min_opt=dc['a_min_opt']
    spec_opt=dc['spec_opt']
    A_opt=dc['A_opt']
    B_opt=dc['B_opt']
    c_opt=dc['c_opt']
    sens_opt=dc['sens_opt']
  else:
    a_min_opt=10**8
    spec_opt=np.random.uniform(0,1,img.shape[2])
    A_opt,B_opt,c_opt=zone_choix(img.shape[:2])
    sens_opt=1
  for k in range(K):
    A,B,c=zone_choix(img.shape[:2])
    spec = np.random.uniform(0,1,img.shape[2])
    assert min(spec)>0, (min(spec))
    A,B,c,spec = fait_choix(A,B,c,spec,A_opt,B_opt,c,spec_opt)   
    img2 = zone_img(img.shape[:2],A,B,c)
    img4 = prg.img_prod(img1,spec)
    assert len(img4[(zone_img(img.shape[:2],A,B,c)==1)&(zone_img(img.shape[:2],A,B,1)==0)])>0, (
      len(img4[(zone_img(img.shape[:2],A,B,c)==1)&(zone_img(img.shape[:2],A,B,1)==0)]),
    )
    min_max = lambda A: (min(A),max(A))
    val_1min,val_1max = min_max(img4[(zone_img(img.shape[:2],A,B,c)==1)&(zone_img(img.shape[:2],A,B,1)==0)])
    val_2min,val_2max = min_max(img4[(zone_img(img.shape[:2],A,B,c)==-1)&(zone_img(img.shape[:2],A,B,1)==0)])
    a_min=min(val_1max-val_2min,val_2max-val_1min)
    # assert a_min>0, (val_1min,val_1max,val_2min,val_2max)
    if a_min<a_min_opt:
      spec_opt=spec; A_opt=A; B_opt=B; c_opt=c; a_min_opt=a_min
      if val_1max-val_2min<val_2max-val_1min:
        sens_opt=-1
      else: 
        sens_opt=1
      prg.save('data_exp5.pkl',['A_opt','B_opt','c_opt','spec_opt','a_min_opt','sens_opt'],[A_opt,B_opt,c_opt,spec_opt,a_min_opt,sens_opt])  
      nom_fichier='data_exp5_sol_{0}.pkl'.format(k)
      prg.save(nom_fichier,['A_opt','B_opt','c_opt','spec_opt','a_min_opt','sens_opt'],[A_opt,B_opt,c_opt,spec_opt,a_min_opt,sens_opt])  
    if val_1min>val_2max+500: 
      return (1,spec,val_1min,val_2max,A_opt,B_opt,c_opt)
    if val_1max<val_2min-500: 
      return (-1,spec,val_1max,val_2min,A_opt,B_opt,c_opt)
    print("k=",k," a_min={0:.2f}".format(a_min)," a_min_opt={0:.2f}".format(a_min_opt))  
  return (sens_opt,spec_opt,val_1min,val_2max,A_opt,B_opt,c_opt)

def Hrho_retrieve(n,call):
  """consulte la base en fonction de la demande
  """
  import numpy as np
  import prg 
  dc_load_53=prg.load('data_exp53.pkl')
  tabRho = dc_load_53['tabRho']
  dcRho  = dc_load_53['dcRho'] 
  tab    = dc_load_53['tab']
  dc     = dc_load_53['dc'] 
  N_max  = tabRho.shape[1]
  def select(n):
    import numpy as np
    test1 = tab[dc["H"],:]< tabRho[dcRho["HM"],n]
    test2 = tab[dc["H"],:]>=tabRho[dcRho["Hm"],n]
    test3 = tab[dc["rho"],:]< tabRho[dcRho["RM"],n]
    test4 = tab[dc["rho"],:]>=tabRho[dcRho["Rm"],n]
    return (np.where(test1&test2&test3&test4))[0]
  assert np.isscalar(n) or type(n)==np.array or type(n)==np.ndarray, (n,type(n))
  if (np.isscalar(n) and 0<=n<=N_max) or ((not np.isscalar(n)) and all(0<=n) and all(n<=N_max)):
    if "Hmoy"==call: 
      # print(f"HM_n={tabRho[dcRho["HM"],n]:.2e} N_max={N_max}")
      # print(n)
      assert np.isscalar(n)
      ind = select(n)
      # print(len(ind),ind)
      assert len(ind)>0, ind
      return True,np.mean(tab[dc["H"],ind])
    elif "rhomoy"==call: 
      assert isscalar(n)
      ind = select(n)
      # print(len(ind),ind)
      assert len(ind)>0, ind
      return True,np.mean(tab[dc["rho"],ind])            
    elif "Hm"==call: 
      return True,tabRho[dcRho["Hm"],n]
    elif "HM"==call: 
      return True,tabRho[dcRho["HM"],n]
    elif "Rm"==call: 
      return True,tabRho[dcRho["Rm"],n]
    elif "RM"==call: 
      return True,tabRho[dcRho["RM"],n]
    else: 
      return False,"call inconnu"
  elif -1==n:
    V_l   = np.zeros(N_max)
    for n in range(N_max):
      ok,V_l[n]=prg.Hrho_retrieve(n,call)
      if not ok:
        return False,N_max
    return True,V_l    
  else: 
    return False,N_max

def img_normalize(img,spec_moy,spec_std):
  """centre et supprime la variance pour le spectre"""
  assert len(spec_moy) == img.shape[2]
  assert len(spec_std) == img.shape[2]
  import numpy as np
  img2 = np.zeros(img.shape)
  for k in range(img.shape[2]): 
    if np.abs(spec_std[k])>1e-10:
      img2[:,:,k]=(img[:,:,k]-spec_moy[k])/spec_std[k]
    else :
      img2[:,:,k]=img[:,:,k]-spec_moy[k]
  return img2

def img_spec(img):
  """computes average variance of spectrum"""
  import numpy as np
  assert len(img.shape) == 3
  moy=np.zeros(img.shape[2])
  std=np.zeros(img.shape[2])
  for k in range(len(moy)): 
    moy[k] = np.sum(img[:,:,k])/img.shape[0]/img.shape[1]
    moy[k] = np.std(img[:,:,k])
  return moy,std

def img_prod(img,spec):
  """ product of hyperspectral image img with spectrum spec yields an image
  """
  import numpy as np
  assert len(img.shape)==3
  assert len(spec) == img.shape[2]
  img2 = np.zeros(img.shape[:2])
  for k in range(len(spec)):
    img2[:,:] += img[:,:,k]*spec[k]/len(spec)
  return img2 

def r90(P):
  """turns 90° around [0,0]"""
  import numpy as np
  return np.array([P[1],-P[0]])

def geo_checkers(M,T):
  import numpy as np
  parite = lambda x,T: 2*(x/T-np.floor(x/T)<1/2)-1
  img=np.zeros((M,M))
  for m in range(M):
    for n in range(M): 
      img[m][n]=parite(m,T)*parite(n,T)
  return img

def geo_angle(M):
  import numpy as np
  return np.angle(M[0]+1j*M[1])

def geo_norm(M):
  import numpy as np
  return np.linalg.vector_norm(M)
  
def fait_choix(A_nv,B_nv,c_nv,spec_nv,A_opt,B_opt,c_opt,spec_opt):
  import numpy as np
  choix = np.random.randint(0,4)
  if 0==choix:
    return A_nv,B_opt,c_opt,spec_opt
  elif 1==choix: 
    return A_opt,B_nv,c_opt,spec_opt
  elif 2==choix:
    return A_opt,B_opt,c_nv,spec_opt
  elif 3==choix:
    ind=np.random.randint(0,len(spec_nv)-1)
    spec=spec_opt
    spec[ind]=spec_nv[ind]
    return A_opt,B_opt,c_opt,spec
  elif 4==choix: 
    choix2 = np.random.randint(0,4)
    if 1==choix2:
      return A_nv,B_nv,c_nv,spec_nv
    else: 
      return A_nv,B_nv,c_nv,spec_opt

def fait_choix2(list_nv,list_opt):
  import numpy as np
  N=len(list_nv)
  choix = np.random.randint(0,N+1)
  if choix<N:
    list        = list_opt
    list[choix] = list_nv[choix]
    return list
  else:
    return list_nv


def val(x):
  """vérifie si x est un numpy array contenant une seule valeur, 
  une seule valeur ou autre chose
  Si c'est autre chose, cela met une erreur.
  Sinon cela renvoie cette unique valeur
  """
  import numpy as np
  if np.isscalar(x):
    return x
  elif type(x)==np.array and 1==len(x):
    return x[0]
  elif type(x)==np.ndarray and 1==len(x):
    return x[0]
  else:
    try:
      print(f"x est de type {type(x)}, de longueur {len(x)}")
      assert False
    except:
      assert False

def vect(x):
  """vérifie si x est un numpy array contenant plusieurs valeurs ou une, 
  mais ce n'est pas un tuple
  Si c'est autre chose, cela met une erreur.
  Sinon cela renvoie ce vecteur sans le tuple autour
  """
  import numpy as np
  if np.isscalar(x):
    assert False
  # elif type(x)==np.array and len(x)>=1:
    # return x
  elif type(x)==np.ndarray and len(x)>=1:
    return x
  elif type(x)==tuple and len(x)==1:
    return x[0]
  else:
    try:
      print(f"x est de type {type(x)}, de longueur {len(x)}")
      assert False
    except:
      assert False

def matrix(X,dim=0,dtype="float"):
  """checks if X is a numpy ndarray or a list of 1D numpy ndarray. 
  returns a 2D numpy ndarray or raises an error if is something else"""
  import numpy as np
  if type(X) == np.ndarray:
    return X
  elif type(X) == list:
    if 0==len(X):
      return np.zeros((dim,0))
    else: 
      assert 1 == len(X[0].shape)
      if "float"==dtype:
        Y=np.zeros((X[0].shape[0],len(X)),dtype=float)
      elif "int" == dtype:
        Y=np.zeros((X[0].shape[0],len(X)),dtype=int)
      else :
        assert False
      for m in range(len(X)):
        Y[:,m]=X[m]
      return Y  
  else:
    assert False    

def is_int(val) -> bool: 
  import numpy as np
  return np.issubdtype(type(val),np.integer)

def is_eq_int(val) -> bool:
  """returns True if  val is approximately equal to an integer"""
  import numpy as np
  if np.issubdtype(type(val),np.integer):
    return True
  elif np.issubdtype(type(val),float):
    return np.abs(int(val)-val) < 1e-10
  else :
    return False
##################################################################################
def cross_correlate(img, mask, padding='valid'):
  """
  The function uses padding to return the required image size.
  Mask is expected to be smaller than or equal to Image by size.
  Mask should have odd-numbered-shapes to do meaningful padding.
  Mask needs not to be square in shape
  Input image and mask should be in grayscale (for simplicity)
  """
  if mask.shape[0]>img.shape[0] or mask.shape[1]>img.shape[1]:
      raise Exception('Mask is bigger than Image!')
  if len(mask.shape)>2 or len(img.shape)>2:
      raise Exception('Please convert inputs to grayscale!')
  
  result_size = [0,0]
  
  # valid padding - returns image of smaller size than the original
  if padding=='valid':
      result_size[0] = img.shape[0] - mask.shape[0] + 1
      result_size[1] = img.shape[1] - mask.shape[1] + 1
      padded = img[:] # to have a common variable during convolution
      
  else:
      pad_size = [0,0] 
      padded_size = [0,0]
      # full padding - returns bigger image than the original
      if padding=='full':
          pad_size[0] = mask.shape[0] - 1
          pad_size[1] = mask.shape[1] - 1
          result_size[0] = img.shape[0] + mask.shape[0] - 1
          result_size[1] = img.shape[1] + mask.shape[1] - 1
      # same padding - returns image of size equal to original image
      if padding=='same':
          pad_size[0] = int((mask.shape[0] - 1)/2)
          pad_size[1] = int((mask.shape[1] - 1)/2)
          result_size[0] = img.shape[0]
          result_size[1] = img.shape[1] 
      padded_size[0] = img.shape[0] + pad_size[0]*2
      padded_size[1] = img.shape[1] + pad_size[1]*2
      # formulate a dummy padded image
      padded = np.zeros(padded_size)
      # pad the input image
      if pad_size[0] == 0:
          padded[:,pad_size[1]:-pad_size[1]] = img[:]
      elif pad_size[1] == 0:
          padded[pad_size[0]:-pad_size[0],:] = img[:]
      else:
          padded[pad_size[0]:-pad_size[0],pad_size[1]:-pad_size[1]] = img[:]
      
  
  # formulate a dummy result
  result = np.zeros(result_size)
  
  # peform cross-correlation
  for r in tqdm(range(result.shape[0])):
      for c in range(result.shape[1]):
          # we are now at result[r][c]
          val = np.multiply(padded[r:r+mask.shape[0],c:c+mask.shape[1]], mask)
          val = np.sum(np.ravel(val))
          result[r][c] = val
  
  return result
                  
def convolve(img, mask, padding='valid'):
  flipped = mask[::-1, ::-1]
  # once mask is double-flipped, convolution resembles cross-correlation
  return cross_correlate(img, flipped, padding)
  
def edge1(img):
  import numpy as np
  import scipy.ndimage as ni
  mask_H,mask_V = np.array([[-1,1]]), np.array([[-1],[1]])
  img2 = ni.convolve(img, mask_H, mode='constant', cval=0.0)
  img3 = ni.convolve(img, mask_V, mode='constant', cval=0.0)
  # img2 = convolve(img,mask,padding='same')  
  img4 = np.abs(img2)+np.abs(img3)<0.01
  img4[:,-1] = 1
  img4[-1,:] = 1
  assert img2.shape==img.shape
  return img4
  
########################################################################################
#verification 
def v_is_int(): 
  assert is_int(1)
  assert not is_int(1.0)
  
def v_edge1(): 
  import numpy as np
  img=np.ones((5,5))
  img2=edge1(img)
  assert (img2==True).all()
  img[2,2]=0
  img2=edge1(img)
  assert (img2==False).sum() == 3
  # print(edge1(img))
def v_en__g():
  import numpy as np
  x=np.abs(np.random.uniform(0,1))
  y=en__g(x)
  assert 0.36<=y<=1
  assert np.abs(y*np.log(y)+y-x)<1e-4

def v_en__g2():
  import numpy as np
  p=np.random.uniform(0,1,5)
  p=p/sum(p)
  epsilon=np.random.uniform(0,1)+np.min(p)
  l=en__g2(p,epsilon)
  q=np.exp(-l*p)/np.sum(np.exp(-l*p))
  objective_function = lambda x: np.sum(p*np.exp(-x*p))-epsilon*np.sum(np.exp(-x*p))
  
  assert np.abs(np.sum(q)-1)<1e-8
  assert np.abs(np.sum(q*p)-epsilon)<1e-8
  
def v_im__shuffle():
  import prg
  import numpy as np
  img,gdr=prg.pine()
  img2 = prg.im__shuffle(img)
  k = np.random.randint(img.shape[2])
  assert np.mean(img[:,:,k]) == np.mean(img2[:,:,k])
  assert not (img[:,:,k] == img2[:,:,k]).all()

def v_zone_fun3():
  import numpy as np
  M=145
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  I=(A+B)/2; rho=geo_norm(B-I)
  theta=geo_angle(A-I)-np.pi/2
  P=np.random.uniform(0,M,2)
  d=geo_norm(I-P)
  assert np.abs(np.cos(geo_angle(A-I)-theta))<1e-5, (geo_angle(A-I),theta,np.sin(geo_angle(A-I)-theta))
  if d>rho:
    assert 0==zone_fun3(I,rho,theta,P), (d,rho,theta,P,I)
  x=np.random.uniform(-1,1)
  Ip=I+x*(B-A)
  assert 0==zone_fun3(I,rho,theta,Ip), (x,rho,theta,I,A)

def v_zone_fun(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  c=np.random.uniform(0,M)
  assert prg.zone_fun(A,B,c,A)
  assert prg.zone_fun(A,B,c,B)
  I=(A+B)/2
  assert prg.zone_fun(A,B,c,I)
  epsi=np.sqrt(np.sum((B-A)*(B-A)))
  u=prg.r90(B-A)/epsi*c
  assert np.abs(np.linalg.vector_norm(u)-c)<1e-8
  assert 1==prg.zone_fun(A,B,c,A+0.99*u+0.01*(B-A))
  assert -1==prg.zone_fun(A,B,c,A-0.99*u+0.01*(B-A))
  assert -1==prg.zone_fun(A,B,c,B-0.99*u-0.01*(B-A))
  assert 1==prg.zone_fun(A,B,c,B+0.99*u-0.01*(B-A))
  assert 0==prg.zone_fun(A,B,c,B+1.01*u-0.01*(B-A))
  
def v_r90():  
  import prg
  plt,np = prg.debut()
  A=np.random.normal(0,1,2)
  B=prg.r90(A)
  assert type(B)==np.ndarray
  assert -1e-8<=np.sum(A*B)<=1e-8
  assert -1e-8<=np.sum(B*B-A*A)<=1e-8
  
def v_zone_img():  
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  c=np.random.uniform(2,M)
  img2=zone_img(img.shape,A,B,c)
  img3=zone_img(img.shape,A,B,1)
  assert np.sum(img2==1)>np.sum(np.abs(img3)==1), (np.sum(img2==1),np.sum(np.abs(img3)==1))
  assert np.sum((img2==1)&(img3==0))>0
  
  
#################################################################################
#save and load
def save(nom_fichier,list_nom_var,list_var):
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
  
def load(nom_fichier):
  """sauvegarde sous format binaire la liste des variables indiquees dans list_var
  le fichier s'appelle nom_fichier
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

def isfile(nom_fichier):
  """rend disponible os.path.isfile 
  True si nom_fichier existe
  """
  import os
  return os.path.isfile(nom_fichier)


def num_exp():
  import inspect as ip
  nom=ip.stack()[1][3]
  if nom[:3]=='ess':
    return int(nom[3:])
  else: 
    assert False

##########################################################
#appel des fonctions 
def verif(): 
  K=10**2
  for k in range(K):
    # v_zone_fun()
    # v_r90()
    # v_zone_img()
    v_zone_fun3()
    print("k=",k)
