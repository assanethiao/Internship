##############################################################
#ess

def ess1():
  """img and ground truth"""
  import prg
  img,gdt = prg.pine()
  plt,np = prg.debut()
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(gdt, cmap='jet')
  ax.axis('off')
  plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()

def ess2():
  import prg
  plt,np = prg.debut()
  plt.close('all')
  t1=np.linspace(-1,1,10**3)
  fig,ax=plt.subplots()
  ax.plot(t1,t1**2,'r:',label='x**2')
  ax.set_xlabel('t')
  # ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()



def ess3(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  c=np.random.uniform(0,M)
  # img2 = np.zeros(img.shape[:2])
  # for i in range(img.shape[0]):
    # for j in range(img.shape[1]):
      # P=np.array([i,j])
      # img2[i][j] = prg.zone_fun(A,B,c,P)
  img2 = zone_img(img.shape[:2],A,B,c)
  img3 = 1-np.abs(zone_img(img.shape[:2],A,B,1))
  #print(zone_ds_img(img.shape[:2],A,B,c))    
      
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow((1+img2)/2, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()

  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(img3, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()


def ess3(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  c=np.random.uniform(0,M)
  # img2 = np.zeros(img.shape[:2])
  # for i in range(img.shape[0]):
    # for j in range(img.shape[1]):
      # P=np.array([i,j])
      # img2[i][j] = prg.zone_fun(A,B,c,P)
  img2 = zone_img(img.shape[:2],A,B,c)
  img3 = 1-np.abs(zone_img(img.shape[:2],A,B,1))
      
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow((1+img2)/2, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()

  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(img3, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()


def ess4(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  A,B,c=zone_choix(img.shape[:2])
  img2 = zone_img(img.shape[:2],A,B,c)
  img3 = 1-np.abs(zone_img(img.shape[:2],A,B,1))
  spec = np.random.uniform(0,1,img.shape[2])
  img4 = prg.img_prod(img,spec)
  min_max = lambda A: (min(A),max(A))
  val_1min,val_1max = min_max(img4[(zone_img(img.shape[:2],A,B,c)==1)&(zone_img(img.shape[:2],A,B,1)==0)])
  val_2min,val_2max = min_max(img4[(zone_img(img.shape[:2],A,B,c)==-1)&(zone_img(img.shape[:2],A,B,1)==0)])
  print(val_1min,val_1max,val_2min,val_2max)
  if val_1min>val_2max: 
    print('regle est zone1 a des valeurs plus eleves que zone2')
  if val_1max<val_2min: 
    print('regle est zone2 a des valeurs plus eleves que zone1')
  
"""  
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow((1+img2)/2, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()

  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(img3, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()
"""

def ess5(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  K=10**6
  sens,spec,val1,val2,A_opt,B_opt,c_opt = zone_find(img,K)

def ess6(): 
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M,N = gdt.shape
  assert M == N, (M,N)
  dc=prg.load('data_exp5.pkl')
  img2 = zone_img(img.shape[:2],dc['A_opt'],dc['B_opt'],dc['c_opt'])
  # plt.close('all')
  fig,ax=plt.subplots()
  plt.imshow((1+img2)/2, cmap='gray')
  ax.axis('off')
  #plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.show()
  


"""profile 
import cProfile
import re
cProfile.run('prg.ess5()')
"""

"""
  # plt.close('all')
  # fig,ax=plt.subplots()
  plt.figure(figsize=(8, 6))
  # plt.imshow((1+img2)/2, cmap='gray')
  # ax.axis('off')
  plt.colorbar(ticks= range(0,16))
  # plt.tight_layout()
  plt.title(f"Indian Pines - Spectral Band ")
  plt.xlabel("X")
  plt.ylabel("Y")
  # fig.show()

  # fig,ax=plt.subplots()
  plt.figure(figsize=(8, 6))
  # plt.imshow(img3, cmap='gray')
  # ax.axis('off')
  plt.colorbar(ticks= range(0,16))
  # plt.tight_layout()
  plt.title(f"Indian Pines - Spectral Band ")
  plt.xlabel("X")
  plt.ylabel("Y")
  # fig.show()
"""

def ess7():
  """dessine la zone"""
  M=145
  import prg
  plt,np=prg.debut()
  A=np.random.uniform(0,M,2)
  B=np.random.uniform(0,M,2)
  theta=np.random.uniform(-np.pi,np.pi)
  theta=0
  I=(A+B)/2; rho=prg.geo_norm(B-I)
  print(prg.zone_ds_img2((M,M),I,rho,theta))
  fun = lambda P: prg.zone_fun3(I,rho,theta,P)
  img2=prg.zone_img_fun((M,M),fun)
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(0.5+0.5*img2, cmap='gray')
  fig.show()

def ess8():
  """dessine la zone"""
  M=145
  plt,np=debut()
  I,rho,theta=zone_choix2((M,M))
  fun = lambda P: zone_fun3(I,rho,theta,P)
  img2=zone_img_fun((M,M),fun)
  print(f"S(img2==1)={np.sum(img2==1)}")
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(0.5+0.5*img2, cmap='gray')
  fig.show()
  
def ess9(): 
  """dessine un jeu d'echec de taille TxT"""
  plt,np=debut()
  M=145; T=15
  parite = lambda x,T: 2*(x/T-np.floor(x/T)<1/2)-1
  
  img=np.zeros((M,M))
  for m in range(M):
    for n in range(M): 
      img[m][n]=parite(m,T)*parite(n,T)
  print(np.mean(img[:][0]))
  plt.close('all')
  fig,ax=plt.subplots()
  # plt.figure(figsize=(8, 6))
  plt.imshow(0.5+0.5*img, cmap='gray')
  fig.show()

def ess10():
  plt,np=debut()
  M=145; T=10
  img1=geo_checkers(M,T)
  I,rho,theta=zone_choix2((M,M))
  fun = lambda P: zone_fun3(I,rho,theta,P)
  img2=zone_img_fun((M,M),fun)
  Np=np.sum(img2==img1); Nm=np.sum(img2+img1==0)
  fun_q = lambda p,q: p*q+(1-p)*(1-q)
  fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np+Nm>0 else 0)
  def fun_Hq(Np,Nm,q): 
    Npq = Np*(1-q)+Nm*q
    Nmq = Np*q+Nm*(1-q)
    return fun_H(Npq,Nmq)
  print(f"Np={Np} et Nm={Nm} H={fun_H(Np,Nm)} H_q={fun_Hq(Np,Nm,0.1)}")
  

def ess11():
  import numpy as np
  img,gdr=pine()
  M=145; K=10
  I,rho,theta=zone_choix2((M,M))
  def fun2(I,rho,theta):
    fun = lambda P: zone_fun3(I,rho,theta,P)
    img2=zone_img_fun((M,M),fun)
    X_tr,Y_tr = ml_dataset(img,img2)
    clf = ml_train1(X_tr,Y_tr)
    Y_te = ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np+Nm>0 else 0)
    return fun_H(Np,Nm)
  I_opt,rho_opt,theta_opt,clf_opt,H=zone_find2(fun2,K)
  
  print(f"Np={Np} et Nm={Nm}  H={fun_H(Np,Nm):.2f}")
  # print(clf.predict(np.array([X_tr[0,:]])))

def ess12():
  import prg
  import numpy as np
  img,gdr=pine()
  M=145; K=10**6
  def fun2(I,rho,theta):
    fun = lambda P: zone_fun3(I,rho,theta,P)
    img2=zone_img_fun((M,M),fun)
    X_tr,Y_tr = ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = ml_train1(X_tr,Y_tr)
    Y_te = ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True)
  if prg.isfile('data_exp12.pkl'):
    dc_load=prg.load('data_exp12.pkl')
    tab=dc_load['tab']
    dc=dc_load['dc']
  else:
    dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"ln":7}
    tab=np.zeros((dc["ln"],0))
  for k in range(K):
    I,rho,theta=zone_choix2((M,M))    
    ligne=np.zeros((dc["ln"],1))
    ligne[dc["Ix"]]=I[0]
    ligne[dc["Iy"]]=I[1]
    ligne[dc["rho"]]=rho
    ligne[dc["theta"]]=theta
    ligne[dc["H"]],ligne[dc["Np"]],ligne[dc["Nm"]],ok = fun2(I,rho,theta)
    H=ligne[dc["H"]][0]
    if ok:
      tab=np.append(tab,ligne,1)
      print(f"k={tab.shape[1]:3d} H={H:5.2f}")
    if (k-K+1)%50==0:
      prg.save('data_exp12.pkl',['tab','dc'],[tab,dc])      

def ess13(): 
  import prg
  plt,np=debut()
  dc_load=prg.load('data_exp12.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']
  ind_rho_gd_15 = (tab[dc['rho']]>=16)
  tab=tab[:,ind_rho_gd_15]
  H_p,H_l=np.histogram(tab[dc['H']], bins='auto',density=True)
  N=len(H_p)
  plt.close('all')
  fig,ax=plt.subplots()
  ax.stairs(H_p,H_l)
  ax.set_xlabel('H')
  # plt.figure(figsize=(8, 6))
  # plt.imshow(gdt, cmap='jet')
  # ax.axis('off')
  # plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  # plt.title(f"Indian Pines - Spectral Band ")
  # plt.xlabel("X")
  # plt.ylabel("Y")
  fig.savefig('./figures/fig_ess13a.png')
  fig.show()

  rho_p,rho_l=np.histogram(tab[dc['rho']],bins=N,density=True)
  assert len(rho_p)==N
  rho2H_moy, rho2H_min = fct(tab[dc['H']],H_l,tab[dc['rho']],rho_l)
  fig,ax=plt.subplots()
  ax.stairs(rho2H_moy,rho_l)
  ax.stairs(rho2H_min,rho_l)
  assert len(H_l)==len(rho2H_moy)+1
  ax.set_xlabel('rho')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess13b.png')
  fig.show()

  Ix_p,Ix_l=np.histogram(tab[dc['Ix']],bins=N,density=True)
  Ix2H_moy, Ix2H_min = fct(tab[dc['H']],H_l,tab[dc['Ix']],Ix_l)
  fig,ax=plt.subplots()
  ax.stairs(Ix2H_moy,Ix_l)
  ax.stairs(Ix2H_min,Ix_l)
  assert len(H_l)==len(Ix2H_moy)+1
  ax.set_xlabel('Ix')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess13c.png')
  fig.show()

  Iy_p,Iy_l=np.histogram(tab[dc['Iy']],bins=N,density=True)
  Iy2H_moy, Iy2H_min = fct(tab[dc['H']],H_l,tab[dc['Iy']],Iy_l)
  fig,ax=plt.subplots()
  ax.stairs(Iy2H_moy,Iy_l)
  ax.stairs(Iy2H_min,Iy_l)
  assert len(H_l)==len(Iy2H_moy)+1
  ax.set_xlabel('Iy')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess13d.png')
  fig.show()

  theta_p,theta_l=np.histogram(tab[dc['theta']],bins=N,density=True)
  theta2H_moy, theta2H_min = fct(tab[dc['H']],H_l,tab[dc['theta']],theta_l)
  fig,ax=plt.subplots()
  ax.stairs(theta2H_moy,theta_l,baseline=None)
  ax.stairs(theta2H_min,theta_l)
  assert len(H_l)==len(theta2H_moy)+1
  ax.set_xlabel('theta')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess13e.png')
  fig.show()


def ess14(): 
  img,gdr=pine()
  def fun2(I,rho,theta):
    fun = lambda P: zone_fun3(I,rho,theta,P)
    img2=zone_img_fun((M,M),fun)
    X_tr,Y_tr = ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = ml_train1(X_tr,Y_tr)
    Y_te = ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True)
  import numpy as np
  import prg
  K=10**6; M=145
  if prg.isfile('data_exp14.pkl'):
    dc_load=prg.load('data_exp14.pkl')
    I_opt=dc_load['I_opt']
    rho_opt=dc_load['rho_opt']
    theta_opt=dc_load['theta_opt']
    H_opt=dc_load['H_opt']
    Np_opt=dc_load['Np_opt']
    Nm_opt=dc_load['Nm_opt']
  else:
    I_opt=np.zeros(2)
    rho_opt=0
    theta_opt=0
    H_opt=10**5
    Np_opt=0
    Nm_opt=0
  for k in range(K):
    I,rho,theta=zone_choix2((M,M))
    liste=fait_choix2([I_opt,rho_opt,theta_opt],[I,rho,theta])
    I=liste[0]; rho=liste[1]; theta=liste[2]
    H,Np,Nm,ok = fun2(I,rho,theta)
    if not ok: 
      continue
    if H<H_opt:
      I_opt=I; rho_opt=rho; theta_opt=theta; Np_opt=Np; Nm_opt=Nm; H_opt=H
      prg.save('data_exp14.pkl',['I_opt','rho_opt','theta_opt','H_opt','Np_opt','Nm_opt'],[I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt])
      nom_fichier=f'data_exp14_sol_{k}.pkl'
      prg.save(nom_fichier,['I_opt','rho_opt','theta_opt','H_opt','Np_opt','Nm_opt'],[I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt])
    if 0==H_opt:
      return I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt
    print(f"k={k} H={H:.2f} H_opt={H_opt:.2f}")     
  return  I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt



def ess15():
  """before ess12"""
  import prg
  import numpy as np
  img,gdr=pine()
  spec_moy,spec_std = img_spec(img)
  img1 = img_normalize(img,spec_moy,spec_std)
  M=145; K=10**6
  def fun2(I,rho,theta):
    fun = lambda P: zone_fun3(I,rho,theta,P)
    img2=zone_img_fun((M,M),fun)
    X_tr,Y_tr = ml_dataset(img1,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    if np.random.randint(0,10)==0:
      gamma_val = ml_train_sel2(X_tr,Y_tr)
    else:
      gamma_val = 3.16e-7
    clf = ml_train2(X_tr,Y_tr,gamma_val)
    Y_te = ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,gamma_val,True)
  if prg.isfile('data_exp15.pkl'):
    dc_load=prg.load('data_exp15.pkl')
    tab=dc_load['tab']
    dc=dc_load['dc']
  else:
    dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
    tab=np.zeros((dc["ln"],0))
  for k in range(K):
    I,rho,theta=zone_choix2((M,M))    
    ligne=np.zeros((dc["ln"],1))
    ligne[dc["Ix"]]=I[0]
    ligne[dc["Iy"]]=I[1]
    ligne[dc["rho"]]=rho
    ligne[dc["theta"]]=theta
    ligne[dc["H"]],ligne[dc["Np"]],ligne[dc["Nm"]],ligne[dc["gamma"]],ok = fun2(I,rho,theta)
    H=ligne[dc["H"]][0]; gamma=ligne[dc["gamma"]][0]
    if ok:
      tab=np.append(tab,ligne,1)
      print(f"k={tab.shape[1]:3d} H={H:5.2f} gamma={gamma:5.2E}")
    if (k-K+1)%50==0:
      prg.save('data_exp15.pkl',['tab','dc'],[tab,dc])      


def ess16(): 
  img,gdr=pine()
  spec_moy,spec_std = img_spec(img)
  img1 = img_normalize(img,spec_moy,spec_std)
  def ess16_fun2(I,rho,theta,gamma_val):
    fun = lambda P: zone_fun3(I,rho,theta,P)
    img2=zone_img_fun((M,M),fun)
    X_tr,Y_tr = ml_dataset(img1,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    if np.isnan(gamma_val):
      gamma_val = ml_train_sel2(X_tr,Y_tr)
    clf = ml_train2(X_tr,Y_tr,gamma_val)
    Y_te = ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return fun_H(Np,Nm),Np,Nm,gamma_val,True
    
  import numpy as np
  import prg
  K=10**6; M=145
  if prg.isfile('data_exp16.pkl'):
    dc_load=prg.load('data_exp16.pkl')
    I_opt=dc_load['I_opt']
    rho_opt=dc_load['rho_opt']
    theta_opt=dc_load['theta_opt']
    H_opt=dc_load['H_opt']
    Np_opt=dc_load['Np_opt']
    Nm_opt=dc_load['Nm_opt']
    gamma_opt=dc_load['gamma_opt']
  else:
    I_opt=np.zeros(2)
    rho_opt=0
    theta_opt=0
    H_opt=10**5
    Np_opt=0
    Nm_opt=0
    gamma_opt=1
  for k in range(K):
    def fun_ok(I,rho,theta): 
      if not rho>=15: 
        return False
      if 'I_opt_l' in dc_load.keys():
        for l in range(len(dc_load['I_opt_l'])/2):
          if not geo_norm(I-dc_load['I_opt_l'][2*l:2*l+1])>=5:
            return False
      else:
        if not geo_norm(I-dc_load['I_opt'])>=5: 
          return False
      return True    
    I,rho,theta=zone_choix3((M,M),fun_ok)
    liste=fait_choix2([I_opt,rho_opt,theta_opt],[I,rho,theta])
    I=liste[0]; rho=liste[1]; theta=liste[2]
    if np.random.randint(0,10)==0:
      gamma_choice=np.nan
    else:
      gamma_choice=gamma_opt
    H,Np,Nm,gamma,ok = ess16_fun2(I,rho,theta,gamma_choice)
    if not ok: 
      continue
    if H<H_opt:
      I_opt=I; rho_opt=rho; theta_opt=theta; Np_opt=Np; Nm_opt=Nm; H_opt=H; gamma_opt=gamma
      if 'I_opt_l' in dc_load.keys():
        I_opt_l=np.append(dc_load['I_opt_l'],I)
      else:   
        I_opt_l=np.append(dc_load['I_opt'],I)
      prg.save('data_exp16.pkl',['I_opt','rho_opt','theta_opt','H_opt','Np_opt','Nm_opt','gamma_opt','I_opt_l'],[I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt,gamma_opt,I_opt_l])
      nom_fichier=f'data_exp16_sol_{k}.pkl'
      prg.save(nom_fichier,['I_opt','rho_opt','theta_opt','H_opt','Np_opt','Nm_opt','gamma_opt','I_opt_l'],[I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt,gamma_opt,I_opt_l])
    if 0==H_opt:
      return I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt,gamma_opt
    print(f"k={k} H={H:.2f} H_opt={H_opt:.2f} gamma_opt={gamma_opt:.2E}")     
  return  I_opt,rho_opt,theta_opt,H_opt,Np_opt,Nm_opt,gamma_opt


def ess17(): 
  import prg
  plt,np=debut()
  dc_load=prg.load('data_exp15.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']
  ind_rho_gd_15 = (tab[dc['rho']]>=16)
  tab=tab[:,ind_rho_gd_15]
  H_p,H_l=np.histogram(tab[dc['H']], bins='auto',density=True)
  N=len(H_p)
  plt.close('all')
  fig,ax=plt.subplots()
  ax.stairs(H_p,H_l)
  ax.set_xlabel('H')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess17a.png')
  fig.show()

  rho_p,rho_l=np.histogram(tab[dc['rho']],bins=N,density=True)
  assert len(rho_p)==N
  rho2H_moy, rho2H_min = fct(tab[dc['H']],H_l,tab[dc['rho']],rho_l)
  fig,ax=plt.subplots()
  ax.stairs(rho2H_moy,rho_l)
  ax.stairs(rho2H_min,rho_l)
  assert len(H_l)==len(rho2H_moy)+1
  ax.set_xlabel('rho')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess17b.png')
  fig.show()

  Ix_p,Ix_l=np.histogram(tab[dc['Ix']],bins=N,density=True)
  Ix2H_moy, Ix2H_min = fct(tab[dc['H']],H_l,tab[dc['Ix']],Ix_l)
  fig,ax=plt.subplots()
  ax.stairs(Ix2H_moy,Ix_l)
  ax.stairs(Ix2H_min,Ix_l)
  assert len(H_l)==len(Ix2H_moy)+1
  ax.set_xlabel('Ix')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess17c.png')
  fig.show()

  Iy_p,Iy_l=np.histogram(tab[dc['Iy']],bins=N,density=True)
  Iy2H_moy, Iy2H_min = fct(tab[dc['H']],H_l,tab[dc['Iy']],Iy_l)
  fig,ax=plt.subplots()
  ax.stairs(Iy2H_moy,Iy_l)
  ax.stairs(Iy2H_min,Iy_l)
  assert len(H_l)==len(Iy2H_moy)+1
  ax.set_xlabel('Iy')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess17d.png')
  fig.show()

  theta_p,theta_l=np.histogram(tab[dc['theta']],bins=N,density=True)
  theta2H_moy, theta2H_min = fct(tab[dc['H']],H_l,tab[dc['theta']],theta_l)
  fig,ax=plt.subplots()
  ax.stairs(theta2H_moy,theta_l,baseline=None)
  ax.stairs(theta2H_min,theta_l)
  assert len(H_l)==len(theta2H_moy)+1
  ax.set_xlabel('theta')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess17e.png')
  fig.show()

def ess18():
  import prg
  plt,np = prg.debut()
  M=145; 
  img,gdr=pine()
  spec_moy,spec_std = img_spec(img)
  img1 = img_normalize(img,spec_moy,spec_std)

  plt.close('all')
  fig,ax=plt.subplots()
  plt.imshow(gdr, cmap='jet')
  ax.axis('off')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess18b.png')
  fig.show()


  #k=1
  def ess18_graph(k):
    nom_fichier=f'data_exp16_sol_{k}.pkl'
    dc_load=prg.load(nom_fichier)
    fun = lambda P: zone_fun3(dc_load['I_opt'],dc_load['rho_opt'],dc_load['theta_opt'],P)
    img2=zone_img_fun((M,M),fun)
  
    fig,ax=plt.subplots()
    plt.imshow(img2, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'./figures/fig_ess18a_{k}.png')
    fig.show()


    X_tr,Y_tr = ml_dataset(img,img2)
    clf=ml_train2(X_tr,Y_tr,dc_load['gamma_opt'])
    img3 = ml_predict(clf,img)
  
  
    fig,ax=plt.subplots()
    plt.imshow(img3, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'./figures/fig_ess18c_{k}.png')
    fig.show()
 
    print(f"rho={dc_load['rho_opt']:.2f} Np={dc_load['Np_opt']} Nm={dc_load['Nm_opt']}") 

  ess18_graph(1)
  ess18_graph(28)
  ess18_graph(155)
  ess18_graph(592)

def ess19():
  """probability distribution of Ix, Iy and theta for H small
  and rho ~= 20
  """
  import prg
  plt,np=debut()
  plt.close('all')
  dc_load=prg.load('data_exp15.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']
  h_min=np.min(tab[dc['H']])
  print(f"H_min={h_min:.2f}")
  is_in = lambda vect,a,b: (a<=vect)&(vect<b)
  ind = is_in(tab[dc['rho']],19.5,20.5)&is_in(tab[dc['H']],0,0.1)
  assert ind.any()
  tab1=tab[:,ind]
  Ix_p,Ix_l=np.histogram(tab1[dc['Ix']], bins='auto',density=True)
  assert np.abs(1-np.sum(Ix_p)*(Ix_l[1]-Ix_l[0]))<1e-5
  assert np.max(tab1[dc['H']])<=0.1
  fig,ax=plt.subplots()
  ax.stairs(Ix_p,Ix_l)
  ax.set_xlabel('Ix')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess19a.png')
  fig.show()

  Iy_p,Iy_l=np.histogram(tab1[dc['Iy']], bins='auto',density=True)
  assert np.abs(1-np.sum(Iy_p)*(Iy_l[1]-Iy_l[0]))<1e-5
  assert np.max(tab1[dc['H']])<=0.1
  fig,ax=plt.subplots()
  ax.stairs(Iy_p,Iy_l)
  ax.set_xlabel('Iy')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess19b.png')
  fig.show()

  theta_p,theta_l=np.histogram(tab1[dc['theta']], bins='auto',density=True)
  assert np.abs(1-np.sum(theta_p)*(theta_l[1]-theta_l[0]))<1e-5
  assert np.max(tab1[dc['H']])<=0.1
  fig,ax=plt.subplots()
  ax.stairs(theta_p,theta_l)
  ax.set_xlabel('theta')
  plt.tight_layout()
  fig.savefig('./figures/fig_ess19c.png')
  fig.show()

def ess20(): 
  """utilisation de MCMC pour sampler I,rho,theta autour de H=0.05
  https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
  """
  def model(theta,age=age):
    a1,a2,a3,p1,p2,p3,T0 = theta
    # model = #your code here
    # return model

def ess21():
  """genere les donnees en supposant que les pixels sont aléatoires"""
  import prg
  import numpy as np
  prg.v_im__shuffle()
  img,gdr=prg.pine()
  spec_moy,spec_std = prg.img_spec(img)
  img1 = prg.img_normalize(img,spec_moy,spec_std)
  img2 = prg.im__shuffle(img1)
  
  K=10**6; M=145  
  def fun2(I,rho,theta):
    gamma_val = 3.16e-7
    fun = lambda P: prg.zone_fun3(I,rho,theta,P)
    img2=prg.zone_img_fun((M,M),fun)
    X_tr,Y_tr = prg.ml_dataset(img1,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = prg.ml_train2(X_tr,Y_tr,gamma_val)
    Y_te = prg.ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,gamma_val,True)
  
  if prg.isfile('data_exp21.pkl'):
    dc_load=prg.load('data_exp21.pkl')
    tab=dc_load['tab']
    dc=dc_load['dc']
  else:
    dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
    tab=np.zeros((dc["ln"],0))
  for k in range(K):
    I,rho,theta=prg.zone_choix2((M,M))    
    ligne=np.zeros((dc["ln"],1))
    ligne[dc["Ix"]]=I[0]
    ligne[dc["Iy"]]=I[1]
    ligne[dc["rho"]]=rho
    ligne[dc["theta"]]=theta
    ligne[dc["H"]],ligne[dc["Np"]],ligne[dc["Nm"]],ligne[dc["gamma"]],ok = fun2(I,rho,theta)
    H=ligne[dc["H"]][0]
    if ok:
      tab=np.append(tab,ligne,1)
      print(f"k={tab.shape[1]:3d} H={H:5.2f} rho={rho:5.2E}")
    if (k-K+1)%50==0:
      prg.save('data_exp21.pkl',['tab','dc'],[tab,dc])      

def ess22():
  import prg
  import numpy as np
  # prg.v_en__g()
  # print(prg.en__g(0),prg.en__g(1))
  for k in range(10**3):
    print(f'k={k}')
    p=np.random.uniform(0,1,5)
    p=p/sum(p)
    epsilon=np.random.uniform(0,1)+np.min(p)
    objective_function = lambda x: np.sum(p*np.exp(-np.exp(x)*p))-epsilon*np.sum(np.exp(-np.exp(x)*p))
    x=np.random.uniform(0,10**5)
    assert np.sum(p*np.exp(-np.exp(x)*p))/np.sum(np.exp(-np.exp(x)*p))<=np.min(p)
  # for k in range(10**3):
    # print(f'k={k}')

    # prg.v_en__g2()
  
def ess23(): 
  """g est convexe"""
  import numpy as np
  for k in range(10**3):
    p=np.random.uniform(0,1,5)
    def g(mu):
      return np.sum(np.exp(mu*p))**(1/mu)
    mu0,mu1=np.random.uniform(0,10**5,2)
    alpha = np.random.uniform(0,1); beta=1-alpha
    assert g(alpha*mu0+beta*mu1)<=alpha*g(mu0)+beta*g(mu1)
  print(f'k={k}')

def ess24(): 
  """g est  minore par exp(min(pi))"""
  import numpy as np
  for k in range(10**5):
    p=np.random.uniform(0,1,5)
    def g(mu):
      return np.sum(np.exp(mu*p))**(1/mu)
    N=np.random.uniform(0,6)  
    mu=np.random.uniform(0,int(10**N))
    assert g(mu)>=np.exp(np.min(p))
    
def ess25(val): 
  """g est n'atteint pas exp(min(pi))"""
  import numpy as np
  # for k in range(10**5):
  p=np.random.uniform(0,1,5)
  def g(mu):
    if np.isscalar(mu):
      return np.sum(np.exp(mu*p))**(1/mu)
    else: 
      res=np.zeros(len(mu))
      for mu_ in range(len(mu)):
        res[mu_]=g(mu[mu_])
      return res
      
  print(g(val)-np.exp(np.min(p)))

def ess26(): 
  """g nouvelle expressioin"""
  import numpy as np
  for k in range(10**5):
    p=np.random.uniform(0,1,5)
    def g_anc(mu):
      if np.isscalar(mu):
        return np.sum(np.exp(mu*p))**(1/mu)
      else: 
        res=np.zeros(len(mu))
        for mu_ in range(len(mu)):
          res[mu_]=g(mu[mu_])
        return res
    def g(mu):
      if np.isscalar(mu):
        pM=np.max(p)
        ln_g=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
        return np.exp(ln_g)
      else: 
        res=np.zeros(len(mu))
        for mu_ in range(len(mu)):
          res[mu_]=g(mu[mu_])
        return res
    N=np.random.uniform(0,6)  
    mu=np.random.uniform(0,10)
    if g(mu)<10**4:
      assert np.abs(g(mu)-g_anc(mu))<1e-7, (np.abs(g(mu)-g_anc(mu)),g(mu),g_anc(mu))

def ess27(): 
  """ln(g) n est pas convexe"""
  import numpy as np
  for k in range(10**3):
    p=np.random.uniform(0,1,5)
    def ln_g(mu):
      if np.isscalar(mu):
        pM=np.max(p)
        ln_g=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
        return ln_g
      else: 
        res=np.zeros(len(mu))
        for mu_ in range(len(mu)):
          res[mu_]=g(mu[mu_])
        return res
    mu0,mu1=np.random.uniform(0,10**5,2)
    alpha = np.random.uniform(0,1); beta=1-alpha
    assert ln_g(alpha*mu0+beta*mu1)<=alpha*ln_g(mu0)+beta*ln_g(mu1)
  print(f'k={k}')

def ess28(): 
  """ln(g) >=pmax"""
  import numpy as np
  for k in range(10**5):
    p=np.random.uniform(0,1,5)
    def ln_g(mu):
      if np.isscalar(mu):
        pM=np.max(p)
        ln_g=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
        return ln_g
      else: 
        res=np.zeros(len(mu))
        for mu_ in range(len(mu)):
          res[mu_]=g(mu[mu_])
        return res
    N=np.random.uniform(0,6)  
    mu=np.random.uniform(0,10)
    assert ln_g(mu)>=np.max(p)
  print(f'k={k}')



def ess29(): 
  """find minimum of ln_g with BFGS proche de pmax
  shows that min ln_g approx max(p)"""
  import numpy as np
  import scipy.optimize as so
  moy=0; K=10**3
  for k in range(K):
    p=np.random.uniform(0,1,5)
    def ln_g(mu):
      pM=np.max(p)
      ln_g=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
      return ln_g
    res=so.minimize(ln_g,1,method='BFGS')
    moy += (res.fun-np.max(p))/K
    assert res.fun>=np.max(p)
  print(f'moy={moy}')


def ess30(): 
  """finds mu given epsilon """
  import numpy as np
  import scipy.optimize as so
  p=np.random.uniform(0,1,5)
  def ln_g(mu):
    if np.isscalar(mu):
      pM=np.max(p)
      val=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
      return val
    else: 
      res=np.zeros(len(mu))
      for mu_ in range(len(mu)):
        res[mu_]=ln_g(mu[mu_])
      return res
  res1=so.minimize(ln_g,1,method='BFGS')
  print(ln_g(res1.x+[0,0.1,0.2, 10,20]))
  # epsilon=res1.fun+np.random.uniform(0,0.2)
  # ln_g_e = lambda mu: (ln_g(mu)-epsilon)**2
  # res2 = so.minimize(ln_g_e,res1.x+1)  
  """
  ln_g_e = lambda mu: ln_g(mu)-epsilon
  # res2 = so.fsolve(ln_g_e,res1.x+1,xtol=1e-12,full_output=True)  
  print(
  if ln_g_e(res1.x+1000)>0:
    res2=so.bisect(ln_g_e,res1.x,res1.x+100,full_output=True)
    print(res1)
    print(res2)
    val=res2[0]
    print(f"mu={val} epsilon={epsilon} ln_g(mu)={ln_g(val)} ln_g_e(mu)={ln_g_e(val)} mu_min={res1.x}")
  else:
    print(ln_g_e(res1.x+100))
  """



def ess31(): 
  """checks if ln_g is decreasing then increasing"""
  import numpy as np
  import scipy.optimize as so
  for k in range(10**8):
    p=np.random.uniform(0,1,7)
    def ln_g(mu):
      if np.isscalar(mu):
        pM=np.max(p)
        val=pM+np.log(np.sum(np.exp(mu*(p-pM))))/mu
        return val
      else: 
        res=np.zeros(len(mu))
        for mu_ in range(len(mu)):
          res[mu_]=ln_g(mu[mu_])
        return res
    N=np.random.uniform(0,6)  
    mu=np.random.uniform(0,int(10**N),4)
    mu.sort()
    if ln_g(mu[1]) < ln_g(mu[2]):
      assert ln_g(mu[2]) <= ln_g(mu[3])
    else:
      assert ln_g(mu[0]) >= ln_g(mu[1])
  print(f'k={k}')

def ess32(): 
  """utilisation de cvxpy"""
  import cvxpy as cp
  import numpy as np
  N=5000
  q = cp.Variable(N)
  p = np.random.uniform(0,1,N)
  p[20:]=0
  epsilon=np.random.uniform(0,2)
  a = np.ones((1,N))
  objective = cp.Maximize(cp.sum(cp.entr(q)))
  constraints = [cp.sum(q) == 1,q>=1e-10,cp.sum(cp.multiply(p,q))<=epsilon]
  problem=cp.Problem(objective,constraints)
  problem.solve()
  print(type(problem.status))
  print("status:", problem.status)
  print("optimal value", problem.value)
  # print("optimal var", q.value)
  if problem.status == 'optimal':
    mu = (np.log(q.value[1])-np.log(q.value[0]))/(p[1]-[0])
    print(f'mu={mu}')
    Z = np.sum(np.exp(mu*p))
    q_th = np.exp(mu*p)/Z
    print(np.abs(q_th-q.value))
    dZdmu = np.sum(p*np.exp(mu*p))
    print(problem.value-(mu-1)*epsilon)
    print(problem.value+np.sum(np.log(q.value)*q.value))
    print(epsilon-np.sum(q.value*p))
    print(f'Z={Z}')
if __name__ == '__main__':
  ess5()
  
def ess33():   
  import prg
  plt,np = prg.debut()
  N,K = 500,50
  ok_l, q_l, epsilon_val_l =  np.zeros(K),np.zeros((K,N)),np.zeros(K)
  p = np.random.uniform(0,1,N)
  # p[10:]=0
  p[0]=0
  epsilon_l=np.linspace(0,1,K)
  for k in range(K):
    ok_l[k],q_l[k,:],epsilon_val_l[k] = prg.en__g3(p,epsilon_l[k])
  # assert all(np.diff(ok_l) >= 0)
  ok_l = (ok_l == 1)
  fig,ax=plt.subplots()
  ax.plot(epsilon_l[ok_l],epsilon_val_l[ok_l])
  ax.set_xlabel('epsilon')
  # ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()

  fig,ax=plt.subplots()
  ax.plot(epsilon_l,ok_l)
  ax.set_xlabel('epsilon')
  # ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()



def ess34(epsilon):   
  import prg
  plt,np = prg.debut()
  plt.close('all')
  N=500
  p = np.random.uniform(0,1,N)
  p[1:100]=0
  p.sort()
  p=p/np.sum(p)
  ok,q,epsilon_val = prg.en__g3(p,epsilon)
  print(f'epsilon={epsilon} epsilon_val={epsilon_val}')
  mu = (np.log(q[-1])-np.log(q[-2]))/(p[-1]-[-2])[0]
  print(f'mu={mu}')
  Z = np.sum(np.exp(mu*p))
  print(f'Z={Z}')
  Hq = np.sum(-np.log(q)*q)
  print(f'Hq={Hq} (mu)*epsilon_val+log(Z)={(-mu)*epsilon_val+np.log(Z)}')
  
  fig,ax=plt.subplots()
  ax.plot(range(N),p/np.max(p),label='p')
  ax.plot(range(N),q/np.max(q),label='q')
  ax.plot(range(N),p*q/np.max(p*q),label='pq')
  ax.plot(range(N),(np.max(p+q)-p-q)/np.max(p+q),label='1-p-q')
  # ax.plot(range(N),p,label='q')
  ax.set_xlabel('n')
  ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()
  print(f'p[12]*q[12]={p[12]*q[12]} max(pq)={np.max(p*q)} max(q)={np.max(q)} max(p)={np.max(p)}')
  
def ess35(): 
  import prg
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  Hmax=np.max(tab[dc['H'],:])
  print(f"H={Hmax}")
  N=10**2
  H_l = np.linspace(0,Hmax,N)
  fig,ax=plt.subplots()
  #a_modifier
  ax.plot(tab[dc['rho'],:],H_l,'.',label='H')
  ax.plot(range(N),q/np.max(q),label='q')
  ax.plot(range(N),p*q/np.max(p*q),label='pq')
  ax.plot(range(N),(np.max(p+q)-p-q)/np.max(p+q),label='1-p-q')
  # ax.plot(range(N),p,label='q')
  ax.set_xlabel('n')
  ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()


def ess36(epsilon):   
  import prg
  plt,np = prg.debut()
  plt.close('all')
  N=500
  p = np.random.uniform(0,1,N)
  p[1:100]=0
  p.sort()
  p=p/np.sum(p)
  ok,q,epsilon_val = prg.en__g3(p,epsilon)
  print(f'epsilon={epsilon} epsilon_val={epsilon_val}')
  mu = (np.log(q[-1])-np.log(q[-2]))/(p[-1]-[-2])[0]
  print(f'mu={mu}')
  Z = np.sum(np.exp(mu*p))
  print(f'Z={Z}')
  Hq = np.sum(-np.log(q)*q)
  print(f'Hq={Hq} (mu)*epsilon_val+log(Z)={(-mu)*epsilon_val+np.log(Z)}')
  print(f"Z-N*np.exp(mu*epsilon)={Z-N*np.exp(mu*epsilon)}")   #proche de 0
  print(f"Z={Z} N*np.exp(mu*epsilon)={N*np.exp(mu*epsilon)}")
  print(f"epsilon={epsilon} (1+mu*epsilon)/(N*np.exp(mu*epsilon))={(1+mu*epsilon)/(N*np.exp(mu*epsilon))}")
  print(f"1/(N-mu)={1/(N-mu)}")


def ess37(): 
  import prg
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  Hmax=np.max(tab[dc['H'],:])
  print(f"H={Hmax}")
  H_p,H_l=np.histogram(tab[dc['H']], bins='auto',density=True)
  print(H_p.shape,H_l.shape)
  ind_max = np.argmax(H_p)
  H_p= H_p[:ind_max]
  H_l= H_l[:ind_max+1]
  epsilon1=0.01
  ok,H_q,epsilon2=prg.en__g3(H_p/np.sum(H_p),epsilon1)
  fig,ax=plt.subplots()
  ax.stairs(H_p,H_l,label='H')
  if ok: 
    H_q = H_q/np.sum(H_q)/(H_l[1]-H_l[0])
    ax.stairs(H_q,H_l,label='Hq')
  print(f"epsilon2 = {epsilon2} et epsilon1={epsilon1}")
  # 
  # ax.plot(range(N),p,label='q')
  ax.set_xlabel('H')
  ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()


def ess37b(): 
  import prg
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  Hmax=np.max(tab[dc['H'],:])
  print(f"H={Hmax}")
  H_p,H_l=np.histogram(tab[dc['H']], bins='auto',density=True)
  # print(H_p.shape,H_l.shape)
  # ind_max = np.argmax(H_p)
  # H_p= H_p[:ind_max]
  # H_l= H_l[:ind_max+1]
  # epsilon1=0.01
  # ok,H_q,epsilon2=prg.en__g3(H_p/np.sum(H_p),epsilon1)
  fig,ax=plt.subplots()
  ax.stairs(H_p,H_l,label='H')
  # if ok: 
    # H_q = H_q/np.sum(H_q)/(H_l[1]-H_l[0])
    # ax.stairs(H_q,H_l,label='Hq')
  # print(f"epsilon2 = {epsilon2} et epsilon1={epsilon1}")
  # 
  # ax.plot(range(N),p,label='q')
  ax.set_xlabel('H')
  ax.legend()
  plt.tight_layout()
  fig.savefig('./figures/fig_ess37b_a.png')
  fig.show()

def ess38(): 
  import cvxpy as cp
  import cvxopt as co
  import numpy as np
  import prg
  plt,np = prg.debut()
  plt.close('all')
  N=2
  q = cp.Variable(N,pos=True)
  qx = cp.Variable(N,pos=True) #q=1/(1+qx)
  p = np.random.uniform(0,1,N)
  p.sort()
  p=p/np.sum(p)
  p=np.array([0.5,0.5])
  objective = cp.Maximize(cp.sum(1/(qx+p+1)))
  constraints = [cp.sum(qx) == 1,qx>=0]
  problem=cp.Problem(objective,constraints)
  problem.solve()
  print(type(problem.status))
  print("status:", problem.status)
  print("optimal value", problem.value)


def ess39(): 
  import numpy as np
  import prg
  def fun(p,q):
    if any(q<0):
      return False,np.nan
    if np.abs(1-np.sum(q))>1e-10:
      return False,np.nan       
    return True, np.sum(q/(p+q))  
  plt,np = prg.debut()
  plt.close('all')
  N,M=5,10**6
  p = np.random.uniform(0,1,N)
  p.sort()
  p=p/np.sum(p)
  q = np.zeros(N)
  j_opt,P_opt,q_opt=np.nan,-np.inf,np.nan
  for n in range(1,N+1): 
    an = np.sum(np.sqrt(p[:n]))
    bn = np.sum(p[:n])
    mu = an/(1+bn)
    q[:n] = np.sqrt(p[:n])/mu - p[:n]
    q[n:] = 0
    if any(q<0): 
      continue
    P = n - an**2/(1+bn)
    ok,P_th = fun(p,q)
    assert ok, (ok,q) 
    assert np.abs(P_th -P)<1e-10
    if P_opt < P : 
      P_opt = P
      j_opt = n
      q_opt = q
  print(f"j_opt = {j_opt} P_opt = {P_opt:.2f}")
  ok,P_th = fun(p,q_opt)
  assert ok ,(ok,q_opt) 
  assert np.abs(P_th-P_opt)<1e-10
  assert np.abs(np.sum(p)-1)<1e-12
  assert all(np.diff(p)>=0)
  for m in range(M):
    q = np.random.uniform(0,1,N)
    q=q/np.sum(q)
    P=np.sum(q/(q+p))
    assert np.abs(1-np.sum(q))<1e-10
    assert P <= P_opt, (P,P_opt,q,q_opt,p)
  
  
def ess40(): 
  import numpy as np
  import prg
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return False,np.nan
    if np.abs(1-np.sum(q))>1e-10:
      print(f"err={np.abs(1-np.sum(q))}")
      return False,np.nan       
    return True, np.sum(q/(p+q))  
  N,M,K=50,10**5,10**3
  for k in range(K):
    p = np.random.uniform(0,1,N)
    P_opt,j_opt,q_opt=prg.en__g4(p)
    ok,P_th = fun(p,q_opt)
    print(f"k={k} j_opt={j_opt} P_opt={P_opt}")
    assert ok , (ok,q_opt,p)
    assert np.abs(P_th-P_opt)<1e-10
    for m in range(M):
      q = np.random.uniform(0,1,N)
      q=q/np.sum(q)
      P=np.sum(q/(q+p))
      assert np.abs(1-np.sum(q))<1e-10
      assert P <= P_opt, (P,P_opt,q,q_opt,p)



def ess41(): 
  import prg
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  Hmax=np.max(tab[dc['H'],:])
  print(f"H={Hmax}")
  H_p,H_l=np.histogram(tab[dc['H']], bins='auto',density=True)
  print(H_p.shape,H_l.shape)
  ind_max = np.argmax(H_p)
  H_p= H_p[:ind_max]
  H_l= H_l[:ind_max+1]
  P,j_opt,H_q=prg.en__g4(H_p/np.sum(H_p))
  print(f"j_opt={j_opt} N={len(H_l)}")
  fig,ax=plt.subplots()
  ax.stairs(H_p,H_l,label='H')
  H_q = H_q/np.sum(H_q)/(H_l[1]-H_l[0])
  H_q=-np.sort(-H_q)
  ax.stairs(H_q,H_l,label='Hq')
  ax.set_xlabel('H')
  ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()
  
  
 
def ess42(): 
  import numpy as np
  import prg
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return False,np.nan
    if np.abs(1-np.sum(q))>1e-10:
      print(f"err={np.abs(1-np.sum(q))}")
      return False,np.nan       
    return True, np.sum(q/(p+q))  
  N,M,K=50,10**5,10**3
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  p.sort()
  q = np.random.uniform(0,1,N)
  q = q/np.sum(q)
  q2=-np.copy(np.sort(-q))
  ok1,P1 = fun(p,q)
  ok2,P2 = fun(p,q2)
  assert ok1 and ok2
  assert P1<=P2




def ess43(): 
  import numpy as np
  import prg
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return False,np.nan
    if np.abs(1-np.sum(q))>1e-10:
      print(f"err={np.abs(1-np.sum(q))}")
      return False,np.nan     
    mn = q + (p-q)*(q>=p)  
    return True, np.sum(mn/(p+q))  
  N,M,K=50,10**5,10**6
  for k in range (K):
    p = np.random.uniform(0,1,N)
    p = p/np.sum(p)
    p.sort()
    q = np.random.uniform(0,1,N)
    q = q/np.sum(q)
    ok1,P1 = fun(p,q)
    ok2,P2 = fun(q,p)
    assert np.abs(P1-P2)<1e-9, (P1,P2)  
    q2=-np.copy(np.sort(-q))
    ok1,P1 = fun(p,q)
    ok2,P2 = fun(p,q2)
    assert ok1 and ok2
    assert P1>=P2, (P1-P2,P1,P2,k,np.min(p),np.min(q))
    # print(f"k={k}")


def ess44(): 
  import numpy as np
  import prg
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return np.sum(mn/(p+q))  
  mu = np.random.normal(0,1)
  N,M,K=10,10**5,10**6
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  p.sort()
  q = np.random.uniform(0,1,N)
  q = q/np.sum(q)
  def J(p,q,mu):
    return fun(p,q) -mu*(sum(q)-1)
  if len(p)==1:  
    if q<p: 
      assert np.abs(fun(p,q)-q/(p+q))<1e-8
    else:
      assert np.abs(fun(p,q)-p/(p+q))<1e-8
  n=np.random.randint(N)  
  epsilon = 1e-4
  q1=q.copy(); q1[n] = q1[n]+ epsilon
  if p[n]<q[n]:
    gra = (-p[n]/(p[n]+q[n])**2)  -mu
  else: 
    gra = (p[n]/(p[n]+q[n])**2)  -mu
  gra = (p[n]/(p[n]+q[n])**2)*(p[n]>=q[n])+(-p[n]/(p[n]+q[n])**2)*(p[n]<q[n])-mu
  # gra = (-p[n]/(p[n]+q[n])**2)  -mu
  # gra = (p[n]/(p[n]+q[n])**2)  -mu
  v=np.abs((J(p,q1,mu) -J(p,q,mu))/epsilon - gra )
  # print(v,gra,(J(p,q1,mu) -J(p,q,mu))/epsilon)
  print(v,n)
  


def ess45(): 
  import numpy as np
  import prg
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return np.sum(mn/(p+q))  
  mu = np.random.normal(0,1)
  N,M,K=10,10**5,10**6
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  p.sort()
  q = np.zeros(N)
  j_opt,P_opt,q_opt=np.nan,-np.inf,np.nan
  for n in range(1,N+1): 
    an1 = np.sum(np.sqrt(p[:n]))
    bn1 = np.sum(p[:n])
    mu1 = (1+bn1)/an1
    q[:n] = np.sqrt(p[:n])*mu1 - p[:n]
    q[n:] = 0
    if any(q[:n]<p[:n]):
      print(f'pb n={n}')
      continue
    if any(q<0): 
      continue
    # P = n - an**2/(1+bn)
    P = fun(p,q)
    # assert ok, (ok,q) 
    # assert np.abs(P_th -P)<1e-10
    if P_opt < P : 
      P_opt = P
      j_opt = n
      q_opt = q
  print(f"j_opt = {j_opt} P_opt = {P_opt:.2f}")
  
  

def ess46():   
  import prg
  plt,np = prg.debut()
  N,M=20,10**6
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return np.sum(mn/(p+q))  
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  #p[0]=0
  p.sort()
  q = np.zeros(N)
  P_opt,q_opt=np.inf,np.nan
  for m in range(M): 
    q = np.random.uniform(0,1,N)
    if np.random.uniform(0,1)<0.03:
      q[1:]=0
      q[0]=1
    q = q/np.sum(q)
    q= -np.sort(-q)
    P=fun(p,q)
    if P_opt>P:
      P_opt=P
      q_opt=q
      print(f"P_opt={P_opt:.4f} m={m}")    
  print(f"P_opt={P_opt:.4f}")    
  plt.close('all')  
  fig,ax=plt.subplots()
  ax.plot(np.arange(N),p,label='p')
  ax.plot(np.arange(N),q_opt,label='q')
  ax.legend()
  plt.tight_layout()
  fig.show()

def ess47():   
  import prg
  plt,np = prg.debut()
  N,M=15,10**5
  def fun(p,q):
    import numpy as np
    import scipy.stats as ss
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return ss.entropy(mn/(p+q))
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  p[0:1]=0
  p.sort()
  q = np.zeros(N)
  P_opt,q_opt=-np.inf,np.nan
  for m in range(M): 
    q = np.random.uniform(0,1,N)
    if np.random.uniform(0,1)<0.5:
      a=np.random.randint(N)
      q[:a]=0
    q = q/np.sum(q)
    q= -np.sort(-q)
    P=fun(p,q)
    if P_opt<P:
      P_opt=P
      q_opt=q
      print(f"P_opt={P_opt:.4f} m={m}")    
  print(f"P_opt={P_opt:.2f}")    
  plt.close('all')  
  fig,ax=plt.subplots()
  ax.plot(np.arange(N),p,label='p')
  ax.plot(np.arange(N),q_opt,label='q')
  ax.legend()
  plt.tight_layout()
  fig.show()

def ess48():   
  import prg
  plt,np = prg.debut()
  N,M=15,10**4
  def fun(p,q):
    import numpy as np
    import scipy.stats as ss
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return ss.entropy(1-mn/(p+q))
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  p.sort()
  print(f"min(p)={np.min(p):5f}")
  q = np.zeros(N)
  P_opt,q_opt=-np.inf,np.nan
  for m in range(M): 
    q = np.random.uniform(0,1,N)
    if np.random.uniform(0,1)<0.5:
      a=np.random.randint(N)
      q[:a]=0
    q = q/np.sum(q)
    q= -np.sort(-q)
    P=fun(p,q)
    if P_opt<P:
      P_opt=P
      q_opt=q
      print(f"P_opt={P_opt:.4f} m={m}")    
  print(f"P_opt={P_opt:.2f}")    
  plt.close('all')  
  fig,ax=plt.subplots()
  ax.plot(np.arange(N),p,label='p')
  ax.plot(np.arange(N),q_opt,label='q')
  ax.legend()
  plt.tight_layout()
  fig.show()



def ess49():   
  import prg
  plt,np = prg.debut()
  N,M=15,10**5
  def fun(p,q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    mn = q + (p-q)*(q>=p)  
    return np.sum(p*mn/(p+q))  
  p = np.random.uniform(0,1,N)
  p = p/np.sum(p)
  #p[0]=0
  p.sort()
  print(f"min(p)={np.min(p):5f}")
  q = np.zeros(N)
  P_opt,q_opt=np.inf,np.nan
  for m in range(M): 
    q = np.random.uniform(0,1,N)
    if np.random.uniform(0,1)<0.03:
      q[1:]=0
      q[0]=1
    q = q/np.sum(q)
    q= -np.sort(-q)
    P=fun(p,q)
    if P_opt>P:
      P_opt=P
      q_opt=q
      print(f"P_opt={P_opt:.4f} m={m}")    
  print(f"P_opt={P_opt:.4f}")    
  plt.close('all')  
  fig,ax=plt.subplots()
  ax.plot(np.arange(N),p,label='p')
  ax.plot(np.arange(N),q_opt,label='q')
  ax.legend()
  plt.tight_layout()
  fig.show()

def ess50(): 
  import prg
  from scipy import optimize as so
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  H=tab[dc['H'],:]
  counts,bins=np.histogram(H,bins=int(np.sqrt(len(H))))
  H_amax=(bins[np.argmax(counts)]+bins[np.argmax(counts)+1])/2
  ind_H_amax=np.where(H<H_amax)[0]
  H2=tab[dc['H'],ind_H_amax]
  rho2=tab[dc['rho'],ind_H_amax]
  H3=(H2-np.mean(H2))/np.std(H2)
  rho3=(rho2-np.mean(rho2))/np.std(rho2)
  def fun(theta):
    import numpy as np
    from numpy import linalg as nl
    a,b=theta[0],theta[1]
    Z=a*H2+b*rho2
    Z.sort()
    counts,bins=np.histogram(Z,bins=int(np.sqrt(len(Z))))
    rank=np.argsort(counts)
    return nl.norm(rank-range(len(rank)))
  res = so.dual_annealing(fun,bounds=[(0,5),(-5,5)])  
  print(res)
  """
 message: ['Maximum number of iteration reached']
 success: True
  status: 0
     fun: 70.52659073002182
       x: [ 2.629e-01  4.391e+00]
     nit: 1000
    nfev: 4022
    njev: 7
    nhev: 0
  """
  
def ess51():
  import prg
  from scipy import optimize as so
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  # Hmax=np.max(tab[dc['H'],:])  
  H=tab[dc['H'],:]
  counts,bins=np.histogram(H,bins=int(np.sqrt(len(H))))
  H_amax=(bins[np.argmax(counts)]+bins[np.argmax(counts)+1])/2
  ind_H_amax=np.where(H<H_amax)[0]
  H2=tab[dc['H'],ind_H_amax]
  rho2=tab[dc['rho'],ind_H_amax]
  print(f"Hmax={H_amax}")
  a=2.26
  b=-2.396e-04
  b=-0.01
  a,b = 0,-1
  Z=a*H2+b*rho2
  Z.sort()
  counts,bins=np.histogram(Z,bins=int(np.sqrt(len(Z))))
  plt.close('all')  
  fig,ax=plt.subplots()
  ax.stairs(counts,bins)
  plt.tight_layout()
  fig.show()


def ess52(): 
  import prg
  from scipy import optimize as so
  plt,np = prg.debut()
  plt.close('all')
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  H=tab[dc['H'],:]
  counts,bins=np.histogram(H,bins=int(np.sqrt(len(H))))
  H_amax=(bins[np.argmax(counts)]+bins[np.argmax(counts)+1])/2
  ind_H_amax=np.where(H<H_amax)[0]
  H2=tab[dc['H'],ind_H_amax]
  rho2=tab[dc['rho'],ind_H_amax]
  H3=(H2-np.mean(H2))/np.std(H2)
  rho3=(rho2-np.mean(rho2))/np.std(rho2)
  def fun(theta):
    import numpy as np
    from numpy import linalg as nl
    a,b=theta[0],theta[1]
    Z=a*H2+b*rho2
    Z.sort()
    counts,bins=np.histogram(Z,bins=int(np.sqrt(len(Z))))
    rank=np.argsort(counts)
    return nl.norm(rank-range(len(rank)))
  res = so.dual_annealing(fun,bounds=[(0,5),(-5,5)])  
  print(res)
  """
 message: ['Maximum number of iteration reached']
 success: True
  status: 0
     fun: 70.52659073002182
       x: [ 2.629e-01  4.391e+00]
     nit: 1000
    nfev: 4022
    njev: 7
    nhev: 0
  """
  
def ess53():
  import prg 
  import numpy as np
  dc_load=prg.load('data_exp21.pkl')
  tab=dc_load['tab']
  dc=dc_load['dc']  #{"Ix":0,"Iy":1,"rho":2,"theta":3,"H":4,"Np":5,"Nm":6,"gamma":7,"ln":8}
  H=tab[dc['H'],:]
  counts,bins=np.histogram(H,bins=int(np.sqrt(len(H))))
  H_amax=(bins[np.argmax(counts)]+bins[np.argmax(counts)+1])/2
  rhom,rhoM = np.min(tab[dc["rho"],:]),np.max(tab[dc["rho"],:])
  ind_H_amax=np.where(H<H_amax)[0]
  tab=(tab[:,ind_H_amax]).copy()
  H=tab[dc["H"],:]
  def histo(H,val): 
    assert type(val) == int
    counts,bins = np.histogram(H,bins=val)
    # print(f"len(H)={len(H)} len(counts)={len(counts)}")
    return counts,bins
  counts_H,bins_H = histo(H,int(np.sqrt(np.sqrt(len(H)))))
  dcH={"Hm":0,"HM":1,"ln":2}
  tabH=np.zeros((dcH["ln"],len(counts)))
  dcRho={"Rm":0,"RM":1,"Hm":2,"HM":3,"ln":4}
  tabRho=np.zeros((dcRho["ln"],0))
  for H_ in range(len(counts_H)):
    Hm,HM=bins_H[H_],bins_H[H_+1]
    tabH[dcH["Hm"],H_],tabH[dcH["HM"],H_]=Hm,HM
    cond = lambda h: (bins_H[H_]<=h)&(h<bins_H[H_+1])
    # print(np.where(cond(tab[dc["H"],:])))
    ind=prg.vect(np.where(cond(tab[dc["H"],:])))
    if 0==len(ind):
      continue
    elif len(ind)<10:
      rho=tab[dc["rho"],ind]
      assert len(rho)>=1, (len(rho),ind)
      ligneRho=np.zeros((dcRho["ln"],1))
      ligneRho[dcRho["Hm"]],ligneRho[dcRho["HM"]]=Hm,HM
      ligneRho[dcRho["Rm"]],ligneRho[dcRho["RM"]]=rhom,rhoM
    else:
      counts,bins= histo(rho,int(np.sqrt(len(rho))))
      for R_ in range(len(counts)):
        ligneRho=np.zeros((dcRho["ln"],1))
        ligneRho[dcRho["Hm"]],ligneRho[dcRho["HM"]]=Hm,HM
        Rm,RM = bins[R_],bins[R_+1]
        ligneRho[dcRho["Rm"]],ligneRho[dcRho["RM"]]=Rm,RM
        tabRho=np.append(tabRho,ligneRho,1)
  prg.save('data_exp53.pkl',['tab','dc','tabH','dcH','tabRho','dcRho'],[tab,dc,tabH,dcH,tabRho,dcRho])      
  ok,N_max=prg.Hrho_retrieve(-1,"Hmoy");  assert not ok
  H_l   = np.zeros(N_max)
  for n in range(N_max):
    ok,H_l[n]=prg.Hrho_retrieve(n,"Hmoy"); assert ok
  ind_tabRho=np.argsort(H_l)
  tabRho = (tabRho[:,ind_tabRho]).copy()
  prg.save('data_exp53.pkl',['tab','dc','tabH','dcH','tabRho','dcRho'],[tab,dc,tabH,dcH,tabRho,dcRho])      

def ess54(): 
  import prg
  plt,np = prg.debut()
  plt.close('all')
  ok,N_max=prg.Hrho_retrieve(-1,"Hmoy");  assert not ok
  # ok,N_max=prg.Hrho_retrieve(0,"Hmoy")
  H_l   = np.zeros(N_max)
  rho_l = np.zeros(N_max)
  for n in range(N_max):
    ok,H_l[n]=prg.Hrho_retrieve(n,"Hmoy"); assert ok
    ok,rho_l[n]=prg.Hrho_retrieve(n,"rhomoy"); assert ok
  fig,ax=plt.subplots()
  ax.plot(np.arange(N_max),H_l,label='H')
  ax.set_xlabel('n')
  # ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()
  fig,ax=plt.subplots()
  ax.plot(np.arange(N_max),rho_l,label='rho')
  ax.set_xlabel('n')
  plt.tight_layout()
  fig.show()

def ess55(): 
  import prg
  K=10**2
  plt,np = prg.debut()
  from scipy import optimize as so
  ok,p = prg.Hrho_retrieve(-1,"Hmoy"); assert ok
  p=p/np.sum(p)
  def fun(q):
    import numpy as np
    if any(q<0):
      print(f"min(q)={np.min(q)}")
      return np.nan
    q=q/np.sum(q)  
    q=-np.sort(-q)
    mn = q + (p-q)*(q>=p)  
    return np.sum(p*mn/(p+q))  
  res = so.dual_annealing(fun,bounds=[(0,1)]*len(p))    
  print(res)
  for k in range(K):
    q0=res.x
    res = so.dual_annealing(fun,bounds=[(0,1)]*len(p),x0=q0)
    print(f"k={k} res.fun={res.fun:.6e} res.x[0]={np.max(res.x)/np.sum(res.x):.4f}")
  print(res)  
  q=res.x
  q=q/np.sum(q)
  q=-np.sort(-q)
  fig,ax=plt.subplots()
  ax.plot(np.arange(len(p)),p,label='p')
  ax.plot(np.arange(len(q)),q,label='q')
  ax.set_xlabel('n')
  # ax.legend()
  plt.tight_layout()
  #fig.savefig('./figures/fig_TP2_fig10a.png')
  fig.show()
  prg.save('data_exp55.pkl',['p','q'],[p,q])      

def ess55b(): 
  import prg 
  plt,np = prg.debut()
  dc_load=prg.load('data_exp55.pkl')      
  p,q = dc_load['p'],dc_load['q']
  H_l = np.zeros(len(p))
  for n in range(len(p)):
    ok,H   = prg.Hrho_retrieve(n,"Hmoy"); assert ok
    H_l[n] = H
  fig,ax=plt.subplots()
  ax.plot(H_l,p,label='p')
  ax.plot(H_l,q,label='q')
  ax.set_xlabel('H')
  # ax.legend()
  plt.tight_layout()
  fig.savefig('./figures/fig_ess55b_a.png')
  fig.show()
  
  
def ess56(): 
  import prg
  plt,np = prg.debut()
  dc_load=prg.load('data_exp55.pkl')
  p=dc_load['p']
  q=dc_load['q']
  print(f"len(q)={len(q)}, q[:4]={q[:4]} max(q[4:])={np.max(q[4:])}")
  ok,Hm = prg.Hrho_retrieve(np.arange(5),"Hm"); assert ok
  ok,HM = prg.Hrho_retrieve(np.arange(5),"HM"); assert ok
  ok,Rm = prg.Hrho_retrieve(np.arange(5),"Rm"); assert ok
  ok,RM = prg.Hrho_retrieve(np.arange(5),"RM"); assert ok
  for n in range(5):
    print(f"Hm={Hm[n]:.3f} HM={HM[n]:.3f} Rm={Rm[n]:.3f} RM={RM[n]:.3f} ")
  """pour rho entre 17.6 et 19.8, il faut que H soit une probabilité uniforme entre 0 et 0.068
  pour rho entre 19.8 et 22.0, il faut que H soit une loi uniforme entre 0 et 0.045
  pour rho entre 15.4 et 17.6, il faut que H soit une loi uniforme entre 0 et 0.045"""
  
def ess57():
  import prg
  plt,np=prg.debut()
  M=145
  Rm_l,RM_l,Hm_l,HM_l=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
  Rm_l[0],RM_l[0],Hm_l[0],HM_l[0]=15.4,17.6,0,0.045
  Rm_l[1],RM_l[1],Hm_l[1],HM_l[1]=17.6,19.8,0,0.068
  Rm_l[2],RM_l[2],Hm_l[2],HM_l[2]=19.8,22.0,0,0.045
  img,gdr=prg.pine()
  def fun2(I,rho,theta):
    fun = lambda P: prg.zone_fun3(I,rho,theta,P)
    img2=prg.zone_img_fun((M,M),fun)
    X_tr,Y_tr = prg.ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = prg.ml_train1(X_tr,Y_tr)
    Y_te = prg.ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True)
  dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"Np":4,"Nm":5,"H":6,"Rm":7,"RM":8,"Hm":9,"HM":10,"ln":11}  
  tab=np.zeros((dc["ln"],0))
  k=0
  while(True):
    exp=np.random.randint(3)
    Rm,RM,Hm,HM=Rm_l[exp],RM_l[exp],Hm_l[exp],HM_l[exp]
    I,rho,theta=prg.zone_choix2((M,M))
    if not rho>=Rm:
      continue
    if not rho<=RM:
      continue
    H,Np,Nm,ok=fun2(I,rho,theta)  
    if not ok: 
      continue
    if not H>=Hm:
      continue
    k = k+1
    if 0==k%200:
      print(f"k={k}, H={H}")
    if not H<=HM:
      continue
    ligne=np.zeros((dc["ln"],1))  
    ligne[dc["Ix"]],ligne[dc["Iy"]]     = I[0],I[1]
    ligne[dc["rho"]],ligne[dc["theta"]] = rho ,theta
    ligne[dc["H"]]                      = H
    ligne[dc["Np"]], ligne[dc["Nm"]]    = Np, Nm
    ligne[dc["Hm"]], ligne[dc["HM"]]    = Hm, HM
    ligne[dc["Rm"]], ligne[dc["RM"]]    = Rm, RM
    tab=np.append(tab,ligne,1)
    #print(ligne)
    print(f"l={tab.shape[1]} H={H} k={k}")
    prg.save('data_exp57b.pkl',['tab','dc'],[tab,dc])        
  print(f"rho={rho:.3f} theta={theta:.2f} H={H:.3f} Np={Np} Np={Nm} ")

def ess58():  
  import prg
  plt,np = prg.debut()
  img,gdt = prg.pine()
  M=145 
  def ess58_graph(k):
    nom_fichier=f"data_exp57a.pkl"
    dc_load=prg.load(nom_fichier)
    tab,dc=dc_load['tab'],dc_load['dc']
    I=np.array([tab[dc['Ix'],k],tab[dc['Iy'],k]])
    fun = lambda P: prg.zone_fun3(I,tab[dc['rho'],k],tab[dc['theta'],k],P)
    img2=prg.zone_img_fun((M,M),fun)
  
    fig,ax=plt.subplots()
    plt.imshow(img2, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'./figures/fig_ess58a_{k}.png')
    # fig.show()


    X_tr,Y_tr = prg.ml_dataset(img,img2)
    clf       = prg.ml_train1(X_tr,Y_tr)
    img3      = prg.ml_predict(clf,img)
  
  
    fig,ax=plt.subplots()
    plt.imshow(img3, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'./figures/fig_ess18c_{k}.png')
    fig.show()
 
    print(f"rho={tab[dc['rho'],k]:.2f} Np={tab[dc['Np'],k]} Nm={tab[dc['Nm'],k]}") 
    return I,rho,theta

  K=4
  Ix_l,Iy_l,rho_l,theta_l=np.zeros(K),np.zeros(K),np.zeros(K),np.zeros(K)
  for k in range(4):
    I,rho_l[k],theta_l[k]=ess58_graph(k)
    Ix_l[k],Iy_l[k]=I[0],I[1]
    
def ess59():
  import numpy as np
  def ess_ess():
    try: 
      s[0]=s[0]+1
      assert False, (s)
    except:
      print("l'instruction s[0]=s[0]+1 provoque une erreur")
      print("parce que s est une liste definie a l'exterieur de la fonction")
      pass
    y=s
    print(f"s={s} y={y}") 
    print("l'instruction y=s realise une copie par reference et non par valeur")
    y[0]=y[0]+1
    print(f"s={s} y={y}") 
    assert y==s
    print("l'instruction y[0]=y[0]+1 ne provoque plus d'erreur car y est maintenant defini dans la fonction")
    print("mais la modification de y entraine une modification de s")    
  s=[3]
  ess_ess()    
  assert not s==[3]
  
  
def ess60():
  import prg
  plt,np=prg.debut()
  nom_pkl = lambda val : 'data_exp'+str(val)+'b.pkl'
  print(nom_pkl(prg.num_exp()))
  M=145
  Rm_l,RM_l,Hm_l,HM_l=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
  Rm_l[0],RM_l[0],Hm_l[0],HM_l[0]=15.4,17.6,0,0.045
  Rm_l[1],RM_l[1],Hm_l[1],HM_l[1]=17.6,19.8,0,0.068
  Rm_l[2],RM_l[2],Hm_l[2],HM_l[2]=19.8,22.0,0,0.045
  rho_est_ok = lambda rho: np.min(Rm_l)<=rho<=np.max(RM_l)
  def rho_H_est_ok(rho,H):
    for exp in range(3): 
      if Rm_l[exp]<=rho<=RM_l[exp] and Hm_l[exp]<=H<=HM_l[exp]:
        return True,(Rm_l[exp],RM_l[exp],Hm_l[exp],HM_l[exp])
    return False,(1,)
  def choix(): 
    while(True):
      I,rho,theta=prg.zone_choix2((M,M))
      if rho_est_ok(rho):
        break
    return I,rho,theta
  def adapte(passe):
    I_1,rho_1,theta_1=choix()
    facteur=passe[4]
    I_2,rho_2,theta_2=passe[:3]
    I   = I_1+facteur*(I_2-I_1)
    rho = rho_1+facteur*(rho_2-rho_1)
    theta = theta_1+facteur*(theta_2-theta_1)
    return I,rho,theta
  img,gdr=prg.pine()
  def fun2(I,rho,theta):
    fun = lambda P: prg.zone_fun3(I,rho,theta,P)
    img2=prg.zone_img_fun((M,M),fun)
    X_tr,Y_tr = prg.ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = prg.ml_train1(X_tr,Y_tr)
    Y_te = prg.ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True)
  dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"Np":4,"Nm":5,"H":6,"Rm":7,"RM":8,"Hm":9,"HM":10,"ln":11}  
  tab=np.zeros((dc["ln"],0))
  k=0
  I,rho,theta = choix()
  passe=(I,rho,theta,np.inf,0.5) 
  I,rho,theta=adapte(passe)
  while(True):
    H,Np,Nm,ok=fun2(I,rho,theta)
    ok,res=rho_H_est_ok(rho,H)
    if ok:
      ligne=np.zeros((dc["ln"],1))  
      ligne[dc["Ix"]],ligne[dc["Iy"]]     = I[0],I[1]
      ligne[dc["rho"]],ligne[dc["theta"]] = rho ,theta
      ligne[dc["H"]]                      = H
      ligne[dc["Np"]], ligne[dc["Nm"]]    = Np, Nm
      ligne[dc["Hm"]], ligne[dc["HM"]]    = res[2], res[3]
      ligne[dc["Rm"]], ligne[dc["RM"]]    = res[0], res[1]        
      tab=np.append(tab,ligne,1)
      print(f"l={tab.shape[1]} H={H} k={k}")
      prg.save(nom_pkl(prg.num_exp()),['tab','dc'],[tab,dc])        
      I,rho,theta = choix()
      passe=[I,rho,theta,np.inf,0.5]
    elif H<passe[3]:
      # print(f"H mieux: k={k}, H={H:.4f} facteur={passe[4]:.4g}")
      passe=[I,rho,theta,H,0.5]
      I,rho,theta=adapte(passe)
    elif passe[4]<1e-5:
      print(f"redemarre: k={k}, H={passe[3]:.4f} facteur={passe[4]:.4g}")
      I,rho,theta = choix()
      passe=[I,rho,theta,np.inf,0.5]
    else:
      passe[4] = passe[4]*0.99
      I,rho,theta=adapte(passe)
    k = k+1
    if 0==k%2000:
      print(f"k={k}, H={H}")
  print(f"rho={rho:.3f} theta={theta:.2f} H={H:.3f} Np={Np} Np={Nm} ")



def ess61():
  import prg
  plt,np=prg.debut()
  nom_pkl = lambda val : 'data_exp'+str(val)+'a.pkl'
  print(nom_pkl(prg.num_exp()))
  M=145
  Rm_l,RM_l,Hm_l,HM_l=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
  Rm_l[0],RM_l[0],Hm_l[0],HM_l[0]=15.4,17.6,0,0.045
  Rm_l[1],RM_l[1],Hm_l[1],HM_l[1]=17.6,19.8,0,0.068
  Rm_l[2],RM_l[2],Hm_l[2],HM_l[2]=19.8,22.0,0,0.045
  rho_est_ok = lambda rho: np.min(Rm_l)<=rho<=np.max(RM_l)
  def rho_H_est_ok(rho,H):
    for exp in range(3): 
      if Rm_l[exp]<=rho<=RM_l[exp] and Hm_l[exp]<=H<=HM_l[exp]:
        return True,(Rm_l[exp],RM_l[exp],Hm_l[exp],HM_l[exp])
    return False,(1,)
  def choix(): 
    while(True):
      I,rho,theta=prg.zone_choix2((M,M))
      if rho_est_ok(rho):
        break
    return I,rho,theta
  def adapte(passe):
    I_1,rho_1,theta_1=choix()
    facteur=passe[4]
    I_2,rho_2,theta_2=passe[:3]
    I   = I_1+facteur*(I_2-I_1)
    rho = rho_1+facteur*(rho_2-rho_1)
    theta = theta_1+facteur*(theta_2-theta_1)
    return I,rho,theta
  img,gdr=prg.pine()
  def fun2(I,rho,theta):
    fun = lambda P: prg.zone_fun3(I,rho,theta,P)
    img2=prg.zone_img_fun((M,M),fun)
    X_tr,Y_tr = prg.ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = prg.ml_train1(X_tr,Y_tr)
    Y_te = prg.ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True)
  dc={"Ix":0,"Iy":1,"rho":2,"theta":3,"Np":4,"Nm":5,"H":6,"Rm":7,"RM":8,"Hm":9,"HM":10,"ln":11}  
  tab=np.zeros((dc["ln"],0))
  k=0
  I,rho,theta = choix()
  passe=(I,rho,theta,np.inf,0.5) 
  I,rho,theta=adapte(passe)
  while(True):
    H,Np,Nm,ok=fun2(I,rho,theta)
    ok,res=rho_H_est_ok(rho,H)
    if ok:
      ligne=np.zeros((dc["ln"],1))  
      ligne[dc["Ix"]],ligne[dc["Iy"]]     = I[0],I[1]
      ligne[dc["rho"]],ligne[dc["theta"]] = rho ,theta
      ligne[dc["H"]]                      = H
      ligne[dc["Np"]], ligne[dc["Nm"]]    = Np, Nm
      ligne[dc["Hm"]], ligne[dc["HM"]]    = res[2], res[3]
      ligne[dc["Rm"]], ligne[dc["RM"]]    = res[0], res[1]        
      tab=np.append(tab,ligne,1)
      print(f"l={tab.shape[1]} H={H} k={k}")
      prg.save(nom_pkl(prg.num_exp()),['tab','dc'],[tab,dc])        
      I,rho,theta = choix()
      passe=[I,rho,theta,np.inf,0.5]
    elif H<passe[3]:
      # print(f"H mieux: k={k}, H={H:.4f} facteur={passe[4]:.4g}")
      passe=[I,rho,theta,H,0.5]
      I,rho,theta=adapte(passe)
    elif passe[4]<1e-5:
      print(f"redemarre: k={k}, H={passe[3]:.4f} facteur={passe[4]:.4g}")
      I,rho,theta = choix()
      passe=[I,rho,theta,np.inf,0.5]
    else:
      passe[4] = passe[4]*0.99
      I,rho,theta=adapte(passe)
    k = k+1
    if 0==k%2000:
      print(f"k={k}, H={H}")
  print(f"rho={rho:.3f} theta={theta:.2f} H={H:.3f} Np={Np} Np={Nm} ")


def ess62(): 
  import prg
  plt,np=prg.debut()
  plt.close('all')
  file_name_l=['data_exp61a.pkl','data_exp60a.pkl','data_exp60b.pkl','data_exp57a.pkl','data_exp57b.pkl']
  img,gdr=prg.pine()
  M=145
  def fun2(I,rho,theta):
    fun = lambda P: prg.zone_fun3(I,rho,theta,P)
    img2=prg.zone_img_fun((M,M),fun)
    X_tr,Y_tr = prg.ml_dataset(img,img2)
    if (len(Y_tr)<2) or (Y_tr==1).all() or (Y_tr==0).all():
      return 0,0,0,False,np.nan
    assert (Y_tr==1).any()
    assert (Y_tr==0).any(), (Y_tr)
    clf = prg.ml_train1(X_tr,Y_tr)
    Y_te = prg.ml_test1(clf,X_tr)
    Np = np.sum(Y_te==Y_tr); Nm = len(Y_te)-Np
    fun_H = lambda Np,Nm: (-Np/(Np+Nm)*np.log(Np/(Np+Nm))-Nm/(Nm+Np)*np.log(Nm/(Np+Nm)) if Np*Nm>0 else 0)
    return (fun_H(Np,Nm),Np,Nm,True,clf) 
  def ml_pred1(clf,img):
    K=img.shape[2]
    I,J=img.shape[0],img.shape[1]
    img3 = np.zeros((I,J),dtype=int)
    for i in range(I):
      for j in range(J): 
        img3[i,j]=prg.val(clf.predict(np.array([img[i,j,:]])))
    return img3
  img5 = np.zeros((M,M),dtype=float)        
  img_nb = 0  
  cnt = 0
  for file in file_name_l:
    assert prg.isfile(file)
    dc_load=prg.load(file)
    assert 'dc' in dc_load.keys()
    dc=dc_load["dc"]
    assert 'tab' in dc_load.keys()
    tab=dc_load["tab"]
    print(f"tab.shape={tab.shape}")  
    L=tab.shape[1]
    img_nb += L
    for l in range(L):
      I    = np.array([tab[dc["Ix"],l],tab[dc["Iy"],l]])
      rho  = tab[dc["rho"],l]
      theta= tab[dc["theta"],l]
      res = fun2(I,rho,theta)
      assert res[3]==True
      assert res[0]==tab[dc["H"],l]
      assert res[1]==tab[dc["Np"],l]
      assert res[2]==tab[dc["Nm"],l]
      clf = res[4]
      img3 = ml_pred1(clf,img)    
      img4 = prg.edge1(img3)
      img5 += (img4 != 1)
      cnt += 1
      print(f"l={l} sum(img4!=1)={np.mean(img4!=1):.3g} H={res[0]:.3g} rho={rho:.3f} cum_l={cnt} theta={tab[dc["theta"],l]/np.pi:.2f}")
      fig,ax=plt.subplots()
      plt.imshow(img4, cmap='gray')
      ax.axis('off')
      plt.tight_layout()
      fig.savefig(f'./figures/fig_ess62_{l}.png')
      fig.show()
  fig,ax=plt.subplots()
  plt.imshow(1-img5/img_nb, cmap='gray')
  ax.axis('off')
  plt.tight_layout()
  fig.savefig(f'./figures/fig_ess62b.png')
  fig.show()
  fig,ax=plt.subplots()
  plt.imshow(gdr, cmap='jet')
  ax.axis('off')
  plt.colorbar(ticks= range(0,16))
  plt.tight_layout()
  fig.savefig(f'./figures/fig_ess62c.png')
  fig.show()
  print(f"img_nb={img_nb}") 
  
def ess63(): 
  def choix_class():
    """selects randomly the classes corresponding y=1 and those corresponding to y=0"""
    import prg
    plt,np = prg.debut()
    img,gdt=prg.pine()
    C = np.max(gdt)
    C1 = int(C/2)
    rng = np.random.default_rng()
    perm = 1+rng.permutation(C)
    p1 = perm[:C1]; p_1 = perm[C1:]; p0=np.array([0])
    return p1,p_1,p0
  def label2class(label,p1,p_1,p0):
    """transforms multiclass classification problem into a binary classification problem"""
    assert type(p1)  == np.ndarray, (type(p1))
    assert type(p_1) == np.ndarray
    assert type(p0)  == np.ndarray
    if label in p1:
      return 1
    elif label in p_1: 
      return -1
    elif label in p0:
      return 0
    else: 
      assert False
  def pixel_pairs(p1,p_1,p0):
    """collects all relevant pairs of neighboring pixels,
    they have to belong to the same class, have a neighbor
    m,n coordinates 
    C class
    """
    import prg
    img,gdt = prg.pine()
    dc   = {"m":0,"n":1,"C":2,"len":3}
    tab  = []
    for m in range(gdt.shape[0]):
      for n in range(gdt.shape[1]):
        line = np.zeros(dc["len"],dtype=int) 
        if not n+1<gdt.shape[1]:
          continue
        line[dc["m"]],line[dc["n"]] = m,n
        C,Cp = label2class(gdt[m,n],p1,p_1,p0), label2class(gdt[m,n+1],p1,p_1,p0)
        if 0 == C or C!=Cp :
          continue
        assert not C == 0  
        line[dc["C"]]=C
        tab.append(line)    
    return prg.matrix(tab,dim=dc["len"],dtype="int"),dc
  def dataset(tab,dc,r,mode):
    """creates training dataset  (X_tr,Y_tr) and testing dataset (X_te,Y_te)
    r is the relative amount of samples in training set
    mode is "random" means random distribution of samples
    mode is "local" means samples are selected around an anchor
    """
    import prg
    import numpy as np
    def c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,L,N_tr,N_te):
      assert type(L) == np.bool, (type(L))
      assert ind_tr            < N_tr
      assert ind_tr_L1         <= ind_tr
      assert ind_te            < N_te
      assert ind_te_L1         <= ind_te
      assert 1+ind_te+1+ind_tr == ind
      if L:
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        elif ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
      else:     
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr    += 1
        elif ind_tr-ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te    += 1
      assert ind_tr          < N_tr, (ind_tr,N_tr,L,ind_te,N_te)
      assert ind_tr_L1       <= ind_tr
      assert ind_te          < N_te
      assert ind_te_L1       <= ind_te
      assert 1+ind_te+ind_tr == ind
      assert is_tr or ind_te >= 0
      assert not is_tr or ind_tr >= 0
      return ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr
    assert type(tab)    == np.ndarray
    assert tab.shape[0] == dc["len"]
    import numpy.random as nr
    assert prg.is_eq_int(tab[dc["m"],nr.randint(tab.shape[1])])
    assert prg.is_eq_int(tab[dc["n"],nr.randint(tab.shape[1])])
    assert prg.is_int(tab[dc["m"],0])
    assert prg.is_int(tab[dc["n"],0])
    img,gdt        = prg.pine()     
    K              = img.shape[2]
    N_tr,N_te,perm = c_perm(tab,dc,r,mode)
    assert not prg.is_int(perm)
    assert len(perm) == tab.shape[1], (len(perm),perm[0],tab.shape[1])
    X_tr,Y_tr,Z_tr = np.zeros((N_tr,2*K)),np.zeros(N_tr,dtype="bool"),np.zeros((N_tr,2),dtype="int")
    X_te,Y_te,Z_te = np.zeros((N_te,2*K)),np.zeros(N_te,dtype="bool"),np.zeros((N_te,2),dtype="int")
    ind_tr,ind_tr_L1,ind_te,ind_te_L1 = -1,-1,-1,-1
    for ind in range(tab.shape[1]):
      line     = tab[:,perm[ind]]
      assert prg.is_int(line[dc["m"]])
      assert prg.is_int(line[dc["n"]])
      assert prg.is_int(ind)
      assert type(img)==np.ndarray
      assert np.issubdtype(line[dc["m"]],np.integer), (type(line[dc["m"]]))
      assert prg.is_eq_int(line[dc["m"]])
      m,n      = line[dc["m"]], line[dc["n"]]
      feature  = np.concatenate((img[m,n,:],img[m,n+1,:]))
      label    = (1==line[dc["C"]])
      ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr = c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,label,N_tr,N_te)
      assert ind_tr < N_tr
      if is_tr: 
        X_tr[ind_tr,:]   = feature
        Y_tr[ind_tr]     = label
        Z_tr[ind_tr,0]   = m
        Z_tr[ind_tr,1]   = n
      else : 
        X_te[ind_te,:]   = feature
        Y_te[ind_te]     = label
        Z_te[ind_te,0]   = m
        Z_te[ind_te,1]   = n
    assert Y_tr.shape[0] == N_tr
    assert Z_tr.shape[0] == N_tr
    assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1], (N_tr,Y_tr.shape[0],N_te,Y_te.shape[0],tab.shape[1])
    return X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te
  def c_perm(tab,dc,r,mode):
    """creates a permutation depending on r and mode
    """
    import numpy as np
    import numpy.random as nr
    N_tab     = tab.shape[1]
    N_tr,N_te = int(r*N_tab), N_tab-int(r*N_tab)
    if "random" == mode:
      rng  = np.random.default_rng()    
      vect = rng.permutation(N_tab)    
      return N_tr,N_te,vect
    else: 
      assert mode == "local"      
      m0,n0 = nr.randint(M), nr.randint(N)
      dist2 = np.zeros(N_tab)
      for ind in range(N_tab):
        m,n        = tab[dc["m"],ind],tab[dc["n"],ind]
        dist2[ind] = (m-m0)**2+(n-n0)**2
      vect = np.argsort(dist2)
      assert type(vect) == np.ndarray, (type(vect))
      assert prg.is_int(vect[0]), (vect[0])
      return N_tr,N_te,vect
  def OA(Y,Y_app):
    import numpy as np
    return np.mean(Y==Y_app)
  def c_OA(r,mode):
    p1,p_1,p0           = choix_class()
    tab,dc              = pixel_pairs(p1,p_1,p0)  
    X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
    clf                 = prg.ml_train1(X_tr,Y_tr)
    Y_te_pre            = prg.ml_test1(clf,X_te)  
    OA_val              = OA(Y_te,Y_te_pre)
    # print(f"r={r}, mode={mode}, OA={OA_val:.2f}")
    return OA_val

  def c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc):      
    """creates an image showing training (red yellow) and testing (blue green) samples
    """
    import prg
    img,gdt = prg.pine()
    img2    = np.ones((img.shape[0],img.shape[1],3),dtype=float)
    N_tr    = Z_tr.shape[0]
    assert type(Y_tr)      == np.ndarray
    assert len(Y_tr.shape) == 1
    assert Y_tr.shape[0]   == N_tr, (Y_tr.shape[0],N_tr)
    assert Y_te.shape[0]   == Z_te.shape[0]
    assert Z_tr.shape[1]   == 2
    assert tab.shape[0]    == dc["len"]
    assert tab.shape[1]    == N_tr + Z_te.shape[0], (tab.shape[1],N_tr,Z_te.shape[0])
    for ind in range(tab.shape[1]):
      if ind<N_tr and Y_tr[ind]:     
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,1,0
      elif ind<N_tr and not Y_tr[ind]:  
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,0,1
      elif Y_te[ind-N_tr]: 
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,0,0
      else:   
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,1,0
    return img2
  import prg
  plt,np              = prg.debut()  
  plt.close('all')
  p1,p_1,p0           = choix_class()
  img,gdt             = prg.pine()
  M,N                 = gdt.shape[0], gdt.shape[1]
  tab,dc              = pixel_pairs(p1,p_1,p0)  
  assert type(tab)    == np.ndarray
  assert tab.shape[0] == dc["len"], (tab.shape,dc["len"])
  print(f"nb/(M*N)    = {tab.shape[1]/M/N:.3f}")
  
  # r,mode              = 0.3,"random"
  # X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  # assert Y_tr.shape[0]==Z_tr.shape[0]
  # assert Y_tr.shape[0]==X_tr.shape[0]
  # assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1]
  # fig,ax = plt.subplots()
  # img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  # plt.imshow(img2)  
  # plt.tight_layout()
  # fig.savefig('./figures/fig_ess63a.png')  
  # fig.show()
  # plt.imshow(img2)  
  
  # r,mode              = 0.3,"local"
  # X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  # fig,ax = plt.subplots()
  # img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  # plt.imshow(img2)  
  # plt.tight_layout()
  # fig.savefig('./figures/fig_ess63b.png')  
  # fig.show()
  # plt.imshow(img2)  

  r,mode              = 0.3,"random"
  X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  clf = prg.ml_train1(X_tr,Y_tr)
  Y_te_pre = prg.ml_test1(clf,X_te)  
  print(f"OA={OA(Y_te,Y_te_pre):.2f}")
  
  nom_pkl = lambda val : 'data_exp'+str(val)+'e.pkl'
  print(nom_pkl(prg.num_exp()))
  R,K     = 50,250
  r_l     = np.linspace(0.03,0.7,R)
  OA_rd_l = np.zeros((len(r_l),2))
  OA_lo_l = np.zeros((len(r_l),2))
  for r_ in range(len(r_l)):
    rd_l, lo_l = [],[]
    for k in range(K):
      rd_l.append(c_OA(r,"random"))
      lo_l.append(c_OA(r,"local"))
    OA_rd_l[r_,:] = np.mean(rd_l),np.std(rd_l)
    OA_lo_l[r_,:] = np.mean(lo_l),np.std(lo_l)
    print(f"r={r_l[r_]:.2f} OA_rd={OA_rd_l[r_,0]:.2f},{OA_rd_l[r_,1]:.2f} OA_lo={OA_lo_l[r_,0]:.2f},{OA_lo_l[r_,1]:.2f}")
    prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])  
  prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])  
  

def ess64(): 
  def choix_class():
    """selects randomly the classes corresponding y=1 and those corresponding to y=0"""
    import prg
    plt,np = prg.debut()
    img,gdt=prg.pine()
    C = np.max(gdt)
    C1 = int(C/2)
    rng = np.random.default_rng()
    perm = 1+rng.permutation(C)
    p1 = perm[:C1]; p_1 = perm[C1:]; p0=np.array([0])
    return p1,p_1,p0
  def label2class(label,p1,p_1,p0):
    """transforms multiclass classification problem into a binary classification problem"""
    assert type(p1)  == np.ndarray, (type(p1))
    assert type(p_1) == np.ndarray
    assert type(p0)  == np.ndarray
    if label in p1:
      return 1
    elif label in p_1: 
      return -1
    elif label in p0:
      return 0
    else: 
      assert False
  def pixel_pairs(p1,p_1,p0):
    """collects all relevant pairs of neighboring pixels,
    they have to belong to the same class, have a neighbor
    m,n coordinates 
    C class
    """
    import prg
    img,gdt = prg.pine()
    dc   = {"m":0,"n":1,"C":2,"len":3}
    tab  = []
    for m in range(gdt.shape[0]):
      for n in range(gdt.shape[1]):
        line = np.zeros(dc["len"],dtype=int) 
        if not n+1<gdt.shape[1]:
          continue
        line[dc["m"]],line[dc["n"]] = m,n
        C,Cp = label2class(gdt[m,n],p1,p_1,p0), label2class(gdt[m,n+1],p1,p_1,p0)
        if 0 == C or C!=Cp :
          continue
        assert not C == 0  
        line[dc["C"]]=C
        tab.append(line)    
    return prg.matrix(tab,dim=dc["len"],dtype="int"),dc
  def dataset(tab,dc,r,mode):
    """creates training dataset  (X_tr,Y_tr) and testing dataset (X_te,Y_te)
    r is the relative amount of samples in training set
    mode is "random" means random distribution of samples
    mode is "local" means samples are selected around an anchor
    """
    import prg
    import numpy as np
    def c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,L,N_tr,N_te):
      assert type(L) == np.bool, (type(L))
      assert ind_tr            < N_tr
      assert ind_tr_L1         <= ind_tr
      assert ind_te            < N_te
      assert ind_te_L1         <= ind_te
      assert 1+ind_te+1+ind_tr == ind
      if L:
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        elif ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
      else:     
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr    += 1
        elif ind_tr-ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te    += 1
      assert ind_tr          < N_tr, (ind_tr,N_tr,L,ind_te,N_te)
      assert ind_tr_L1       <= ind_tr
      assert ind_te          < N_te
      assert ind_te_L1       <= ind_te
      assert 1+ind_te+ind_tr == ind
      assert is_tr or ind_te >= 0
      assert not is_tr or ind_tr >= 0
      return ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr
    assert type(tab)    == np.ndarray
    assert tab.shape[0] == dc["len"]
    import numpy.random as nr
    assert prg.is_eq_int(tab[dc["m"],nr.randint(tab.shape[1])])
    assert prg.is_eq_int(tab[dc["n"],nr.randint(tab.shape[1])])
    assert prg.is_int(tab[dc["m"],0])
    assert prg.is_int(tab[dc["n"],0])
    img,gdt        = prg.pine()     
    K              = img.shape[2]
    N_tr,N_te,perm = c_perm(tab,dc,r,mode)
    assert not prg.is_int(perm)
    assert len(perm) == tab.shape[1], (len(perm),perm[0],tab.shape[1])
    X_tr,Y_tr,Z_tr = np.zeros((N_tr,2*K)),np.zeros(N_tr,dtype="bool"),np.zeros((N_tr,2),dtype="int")
    X_te,Y_te,Z_te = np.zeros((N_te,2*K)),np.zeros(N_te,dtype="bool"),np.zeros((N_te,2),dtype="int")
    ind_tr,ind_tr_L1,ind_te,ind_te_L1 = -1,-1,-1,-1
    for ind in range(tab.shape[1]):
      line     = tab[:,perm[ind]]
      assert prg.is_int(line[dc["m"]])
      assert prg.is_int(line[dc["n"]])
      assert prg.is_int(ind)
      assert type(img)==np.ndarray
      assert np.issubdtype(line[dc["m"]],np.integer), (type(line[dc["m"]]))
      assert prg.is_eq_int(line[dc["m"]])
      m,n      = line[dc["m"]], line[dc["n"]]
      feature  = np.concatenate((img[m,n,:],img[m,n+1,:]))
      label    = (1==line[dc["C"]])
      ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr = c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,label,N_tr,N_te)
      assert ind_tr < N_tr
      if is_tr: 
        X_tr[ind_tr,:]   = feature
        Y_tr[ind_tr]     = label
        Z_tr[ind_tr,0]   = m
        Z_tr[ind_tr,1]   = n
      else : 
        X_te[ind_te,:]   = feature
        Y_te[ind_te]     = label
        Z_te[ind_te,0]   = m
        Z_te[ind_te,1]   = n
    assert Y_tr.shape[0] == N_tr
    assert Z_tr.shape[0] == N_tr
    assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1], (N_tr,Y_tr.shape[0],N_te,Y_te.shape[0],tab.shape[1])
    return X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te
  def c_perm(tab,dc,r,mode):
    """creates a permutation depending on r and mode
    """
    import numpy as np
    import numpy.random as nr
    N_tab     = tab.shape[1]
    N_tr,N_te = int(r*N_tab), N_tab-int(r*N_tab)
    if "random" == mode:
      rng  = np.random.default_rng()    
      vect = rng.permutation(N_tab)    
      return N_tr,N_te,vect
    else: 
      assert mode == "local"      
      m0,n0 = nr.randint(M), nr.randint(N)
      dist2 = np.zeros(N_tab)
      for ind in range(N_tab):
        m,n        = tab[dc["m"],ind],tab[dc["n"],ind]
        dist2[ind] = (m-m0)**2+(n-n0)**2
      vect = np.argsort(dist2)
      assert type(vect) == np.ndarray, (type(vect))
      assert prg.is_int(vect[0]), (vect[0])
      return N_tr,N_te,vect
  def OA(Y,Y_app):
    import numpy as np
    return np.mean(Y==Y_app)
  def c_OA(r,mode):
    p1,p_1,p0           = choix_class()
    tab,dc              = pixel_pairs(p1,p_1,p0)  
    X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
    clf                 = prg.ml_train1(X_tr,Y_tr)
    Y_te_pre            = prg.ml_test1(clf,X_te)  
    OA_val              = OA(Y_te,Y_te_pre)
    # print(f"r={r}, mode={mode}, OA={OA_val:.2f}")
    return OA_val

  def c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc):      
    """creates an image showing training (red yellow) and testing (blue green) samples
    """
    import prg
    img,gdt = prg.pine()
    img2    = np.ones((img.shape[0],img.shape[1],3),dtype=float)
    N_tr    = Z_tr.shape[0]
    assert type(Y_tr)      == np.ndarray
    assert len(Y_tr.shape) == 1
    assert Y_tr.shape[0]   == N_tr, (Y_tr.shape[0],N_tr)
    assert Y_te.shape[0]   == Z_te.shape[0]
    assert Z_tr.shape[1]   == 2
    assert tab.shape[0]    == dc["len"]
    assert tab.shape[1]    == N_tr + Z_te.shape[0], (tab.shape[1],N_tr,Z_te.shape[0])
    for ind in range(tab.shape[1]):
      if ind<N_tr and Y_tr[ind]:     
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,1,0
      elif ind<N_tr and not Y_tr[ind]:  
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,0,1
      elif Y_te[ind-N_tr]: 
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,0,0
      else:   
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,1,0
    return img2
  import prg
  plt,np              = prg.debut()  
  plt.close('all')
  p1,p_1,p0           = choix_class()
  img,gdt             = prg.pine()
  M,N                 = gdt.shape[0], gdt.shape[1]
  tab,dc              = pixel_pairs(p1,p_1,p0)  
  assert type(tab)    == np.ndarray
  assert tab.shape[0] == dc["len"], (tab.shape,dc["len"])
  print(f"nb/(M*N)    = {tab.shape[1]/M/N:.3f}")
  
  # r,mode              = 0.3,"random"
  # X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  # assert Y_tr.shape[0]==Z_tr.shape[0]
  # assert Y_tr.shape[0]==X_tr.shape[0]
  # assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1]
  # fig,ax = plt.subplots()
  # img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  # plt.imshow(img2)  
  # plt.tight_layout()
  # fig.savefig('./figures/fig_ess63a.png')  
  # fig.show()
  # plt.imshow(img2)  
  
  # r,mode              = 0.3,"local"
  # X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  # fig,ax = plt.subplots()
  # img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  # plt.imshow(img2)  
  # plt.tight_layout()
  # fig.savefig('./figures/fig_ess63b.png')  
  # fig.show()
  # plt.imshow(img2)  

  r,mode              = 0.3,"random"
  X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  clf = prg.ml_train1(X_tr,Y_tr)
  Y_te_pre = prg.ml_test1(clf,X_te)  
  print(f"OA={OA(Y_te,Y_te_pre):.2f}")
  
  nom_pkl = lambda val : 'data_exp'+str(val)+'d.pkl'
  print(nom_pkl(prg.num_exp()))
  R,K     = 50,50
  r_l     = np.linspace(0.03,0.7,R)
  OA_rd_l = np.zeros((len(r_l),2))
  OA_lo_l = np.zeros((len(r_l),2))
  for r_ in range(len(r_l)):
    rd_l, lo_l = [],[]
    for k in range(K):
      rd_l.append(c_OA(r,"random"))
      lo_l.append(c_OA(r,"local"))
    OA_rd_l[r_,:] = np.mean(rd_l),np.std(rd_l)
    OA_lo_l[r_,:] = np.mean(lo_l),np.std(lo_l)
    print(f"r={r_l[r_]:.2f} OA_rd={OA_rd_l[r_,0]:.2f},{OA_rd_l[r_,1]:.2f} OA_lo={OA_lo_l[r_,0]:.2f},{OA_lo_l[r_,1]:.2f}")
    prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])  
  prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])    
  
def ess65(): 
  """classes are no longer randomly chosen, choix_class has a completely different meaning"""
  def choix_class():
    """selects not randomly the classes corresponding y=1 and those corresponding to y=0"""
    import prg
    plt,np = prg.debut()
    img,gdt=prg.pine()
    C = np.max(gdt)
    C1 = int(C/2)
    # rng = np.random.default_rng()
    # perm = 1+rng.permutation(C)
    perm = np.arange(C)+1
    p1 = perm[:C1]; p_1 = perm[C1:]; p0=np.array([0])
    return p1,p_1,p0
  def label2class(label,p1,p_1,p0):
    """transforms multiclass classification problem into a binary classification problem"""
    assert type(p1)  == np.ndarray, (type(p1))
    assert type(p_1) == np.ndarray
    assert type(p0)  == np.ndarray
    if label in p1:
      return 1
    elif label in p_1: 
      return -1
    elif label in p0:
      return 0
    else: 
      assert False
  def pixel_pairs(p1,p_1,p0):
    """collects all relevant pairs of neighboring pixels,
    they have to belong to the same class, have a neighbor
    m,n coordinates 
    C class
    """
    import prg
    img,gdt = prg.pine()
    dc   = {"m":0,"n":1,"C":2,"len":3}
    tab  = []
    for m in range(gdt.shape[0]):
      for n in range(gdt.shape[1]):
        line = np.zeros(dc["len"],dtype=int) 
        if not n+1<gdt.shape[1]:
          continue
        line[dc["m"]],line[dc["n"]] = m,n
        C,Cp = label2class(gdt[m,n],p1,p_1,p0), label2class(gdt[m,n+1],p1,p_1,p0)
        if 0 == C or C!=Cp :
          continue
        assert not C == 0  
        line[dc["C"]]=C
        tab.append(line)    
    return prg.matrix(tab,dim=dc["len"],dtype="int"),dc
  def dataset(tab,dc,r,mode):
    """creates training dataset  (X_tr,Y_tr) and testing dataset (X_te,Y_te)
    r is the relative amount of samples in training set
    mode is "random" means random distribution of samples
    mode is "local" means samples are selected around an anchor
    """
    import prg
    import numpy as np
    def c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,L,N_tr,N_te):
      assert type(L) == np.bool, (type(L))
      assert ind_tr            < N_tr
      assert ind_tr_L1         <= ind_tr
      assert ind_te            < N_te
      assert ind_te_L1         <= ind_te
      assert 1+ind_te+1+ind_tr == ind
      if L:
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        elif ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr_L1 += 1
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te_L1 += 1
          ind_te    += 1
      else:     
        if not ind_tr+1<N_tr:
          is_tr     = False
          ind_te    += 1
        elif not ind_te+1<N_te:   
          is_tr     = True
          ind_tr    += 1
        elif ind_tr-ind_tr_L1 < N_tr/2:
          is_tr     = True
          ind_tr    += 1
        else:
          is_tr     = False
          ind_te    += 1
      assert ind_tr          < N_tr, (ind_tr,N_tr,L,ind_te,N_te)
      assert ind_tr_L1       <= ind_tr
      assert ind_te          < N_te
      assert ind_te_L1       <= ind_te
      assert 1+ind_te+ind_tr == ind
      assert is_tr or ind_te >= 0
      assert not is_tr or ind_tr >= 0
      return ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr
    assert type(tab)    == np.ndarray
    assert tab.shape[0] == dc["len"]
    import numpy.random as nr
    assert prg.is_eq_int(tab[dc["m"],nr.randint(tab.shape[1])])
    assert prg.is_eq_int(tab[dc["n"],nr.randint(tab.shape[1])])
    assert prg.is_int(tab[dc["m"],0])
    assert prg.is_int(tab[dc["n"],0])
    img,gdt        = prg.pine()     
    K              = img.shape[2]
    N_tr,N_te,perm = c_perm(tab,dc,r,mode)
    assert not prg.is_int(perm)
    assert len(perm) == tab.shape[1], (len(perm),perm[0],tab.shape[1])
    X_tr,Y_tr,Z_tr = np.zeros((N_tr,2*K)),np.zeros(N_tr,dtype="bool"),np.zeros((N_tr,2),dtype="int")
    X_te,Y_te,Z_te = np.zeros((N_te,2*K)),np.zeros(N_te,dtype="bool"),np.zeros((N_te,2),dtype="int")
    ind_tr,ind_tr_L1,ind_te,ind_te_L1 = -1,-1,-1,-1
    for ind in range(tab.shape[1]):
      line     = tab[:,perm[ind]]
      assert prg.is_int(line[dc["m"]])
      assert prg.is_int(line[dc["n"]])
      assert prg.is_int(ind)
      assert type(img)==np.ndarray
      assert np.issubdtype(line[dc["m"]],np.integer), (type(line[dc["m"]]))
      assert prg.is_eq_int(line[dc["m"]])
      m,n      = line[dc["m"]], line[dc["n"]]
      feature  = np.concatenate((img[m,n,:],img[m,n+1,:]))
      label    = (1==line[dc["C"]])
      ind_tr,ind_tr_L1,ind_te,ind_te_L1,is_tr = c_ind(ind_tr,ind_tr_L1,ind_te,ind_te_L1,ind,label,N_tr,N_te)
      assert ind_tr < N_tr
      if is_tr: 
        X_tr[ind_tr,:]   = feature
        Y_tr[ind_tr]     = label
        Z_tr[ind_tr,0]   = m
        Z_tr[ind_tr,1]   = n
      else : 
        X_te[ind_te,:]   = feature
        Y_te[ind_te]     = label
        Z_te[ind_te,0]   = m
        Z_te[ind_te,1]   = n
    assert Y_tr.shape[0] == N_tr
    assert Z_tr.shape[0] == N_tr
    assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1], (N_tr,Y_tr.shape[0],N_te,Y_te.shape[0],tab.shape[1])
    return X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te
  def c_perm(tab,dc,r,mode):
    """creates a permutation depending on r and mode
    """
    import numpy as np
    import numpy.random as nr
    N_tab     = tab.shape[1]
    N_tr,N_te = int(r*N_tab), N_tab-int(r*N_tab)
    if "random" == mode:
      rng  = np.random.default_rng()    
      vect = rng.permutation(N_tab)    
      return N_tr,N_te,vect
    else: 
      assert mode == "local"      
      m0,n0 = nr.randint(M), nr.randint(N)
      dist2 = np.zeros(N_tab)
      for ind in range(N_tab):
        m,n        = tab[dc["m"],ind],tab[dc["n"],ind]
        dist2[ind] = (m-m0)**2+(n-n0)**2
      vect = np.argsort(dist2)
      assert type(vect) == np.ndarray, (type(vect))
      assert prg.is_int(vect[0]), (vect[0])
      return N_tr,N_te,vect
  def OA(Y,Y_app):
    import numpy as np
    return np.mean(Y==Y_app)
  def c_OA(r,mode):
    p1,p_1,p0           = choix_class()
    tab,dc              = pixel_pairs(p1,p_1,p0)  
    X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
    clf                 = prg.ml_train1(X_tr,Y_tr)
    Y_te_pre            = prg.ml_test1(clf,X_te)  
    OA_val              = OA(Y_te,Y_te_pre)
    # print(f"r={r}, mode={mode}, OA={OA_val:.2f}")
    return OA_val

  def c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc):      
    """creates an image showing training (red yellow) and testing (blue green) samples
    """
    import prg
    img,gdt = prg.pine()
    img2    = np.ones((img.shape[0],img.shape[1],3),dtype=float)
    N_tr    = Z_tr.shape[0]
    assert type(Y_tr)      == np.ndarray
    assert len(Y_tr.shape) == 1
    assert Y_tr.shape[0]   == N_tr, (Y_tr.shape[0],N_tr)
    assert Y_te.shape[0]   == Z_te.shape[0]
    assert Z_tr.shape[1]   == 2
    assert tab.shape[0]    == dc["len"]
    assert tab.shape[1]    == N_tr + Z_te.shape[0], (tab.shape[1],N_tr,Z_te.shape[0])
    for ind in range(tab.shape[1]):
      if ind<N_tr and Y_tr[ind]:     
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,1,0
      elif ind<N_tr and not Y_tr[ind]:  
        img2[Z_tr[ind,0],Z_tr[ind,1],:] = 0,0,1
      elif Y_te[ind-N_tr]: 
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,0,0
      else:   
        img2[Z_te[ind-N_tr,0],Z_te[ind-N_tr,1],:] = 1,1,0
    return img2
  import prg
  plt,np              = prg.debut()  
  plt.close('all')
  p1,p_1,p0           = choix_class()
  img,gdt             = prg.pine()
  M,N                 = gdt.shape[0], gdt.shape[1]
  tab,dc              = pixel_pairs(p1,p_1,p0)  
  assert type(tab)    == np.ndarray
  assert tab.shape[0] == dc["len"], (tab.shape,dc["len"])
  print(f"nb/(M*N)    = {tab.shape[1]/M/N:.3f}")
  
  r,mode              = 0.3,"random"
  X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  assert Y_tr.shape[0]==Z_tr.shape[0]
  assert Y_tr.shape[0]==X_tr.shape[0]
  assert Y_tr.shape[0]+Y_te.shape[0] == tab.shape[1]
  fig,ax = plt.subplots()
  img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  plt.imshow(img2)  
  plt.tight_layout()
  fig.savefig('./figures/fig_ess65a.png')  
  fig.show()
  plt.imshow(img2)  
  
  r,mode              = 0.3,"local"
  X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  fig,ax = plt.subplots()
  img2 = c_img2(Y_tr,Z_tr,Y_te,Z_te,tab,dc)
  plt.imshow(img2)  
  plt.tight_layout()
  fig.savefig('./figures/fig_ess65b.png')  
  fig.show()
  plt.imshow(img2)  

  r,mode              = 0.3,"random"
  X_tr,Y_tr,Z_tr,X_te,Y_te,Z_te = dataset(tab,dc,r,mode)
  clf = prg.ml_train1(X_tr,Y_tr)
  Y_te_pre = prg.ml_test1(clf,X_te)  
  print(f"OA={OA(Y_te,Y_te_pre):.2f}")
  
  nom_pkl = lambda val : 'data_exp'+str(val)+'e.pkl'
  print(nom_pkl(prg.num_exp()))
  R,K     = 50,50
  r_l     = np.linspace(0.03,0.7,R)
  OA_rd_l = np.zeros((len(r_l),2))
  OA_lo_l = np.zeros((len(r_l),2))
  for r_ in range(len(r_l)):
    rd_l, lo_l = [],[]
    for k in range(K):
      rd_l.append(c_OA(r_l[r_],"random"))
      lo_l.append(c_OA(r_l[r_],"local"))
    OA_rd_l[r_,:] = np.mean(rd_l),np.std(rd_l)
    OA_lo_l[r_,:] = np.mean(lo_l),np.std(lo_l)
    print(f"r={r_l[r_]:.2f} OA_rd={OA_rd_l[r_,0]:.2f},{OA_rd_l[r_,1]:.2f} OA_lo={OA_lo_l[r_,0]:.2f},{OA_lo_l[r_,1]:.2f}")
    prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])  
  prg.save(nom_pkl(prg.num_exp()),['r_l','OA_rd_l','OA_lo_l','r_'],[r_l,OA_rd_l,OA_lo_l,r_])      
  
def ess66():
  import prg 
  dc_load =prg.load('data_exp65e.pkl')
  plt,np  = prg.debut()
  fig,ax  = plt.subplots()
  ax.plot(dc_load["r_l"],dc_load["OA_rd_l"][:,0],'b-',label='rd')
  ax.plot(dc_load["r_l"],dc_load["OA_lo_l"][:,0],'r-',label='lo')
  ax.legend()
  ax.plot(dc_load["r_l"],dc_load["OA_rd_l"][:,0]+dc_load["OA_rd_l"][:,1],'b:')
  ax.plot(dc_load["r_l"],dc_load["OA_rd_l"][:,0]-dc_load["OA_rd_l"][:,1],'b:')
  ax.plot(dc_load["r_l"],dc_load["OA_lo_l"][:,0]+dc_load["OA_lo_l"][:,1],'r:')
  ax.plot(dc_load["r_l"],dc_load["OA_lo_l"][:,0]-dc_load["OA_lo_l"][:,1],'r:')
  ax.set_xlabel('r')
  ax.set_ylabel('OA')
  plt.tight_layout()
  plt.savefig('./figures/fig_ess66a.png')
  fig.show()