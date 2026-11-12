"""Compléments TP4 : génération des cas, boucle, MAP et BCR.
Unités SI ; X est de taille (6,N), observations à t=1,...,720 s.
Voir lancer_tp.py et README.md.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist

SIGMA0 = np.array([3000.,3000.,500.,5.,5.,5.])
QSTD = np.array([.5,.5,.1,.1,.1,.01])
V = 300/3.6
DT = 1.
T = 720
F = np.eye(6); F[:3,3:] = np.eye(3)*DT
R = 30.**2

class Terrain:
    """Convention identique au code fourni : noeud (j,i) à (i*dx,j*dy).
    Ne prolonge pas artificiellement les bords. NaN => vraisemblance nulle.
    """
    def __init__(self, path=Path(__file__).with_name('carte.mat')):
        m=loadmat(path)
        self.z=np.asarray(m['h_MNT'],float)
        self.dx=float(m['pasx_reel'].item()); self.dy=float(m['pasy_reel'].item())
        self.ny,self.nx=self.z.shape
    def evaluate(self,x,y,gradient=False):
        x,y=np.broadcast_arrays(x,y)
        a=x/self.dx; b=y/self.dy
        valid=(a>=0)&(a<self.nx-1)&(b>=0)&(b<self.ny-1)
        i=np.floor(np.clip(a,0,self.nx-2)).astype(int)
        j=np.floor(np.clip(b,0,self.ny-2)).astype(int)
        u=a-i; v=b-j
        z00=self.z[j,i];z10=self.z[j,i+1];z01=self.z[j+1,i];z11=self.z[j+1,i+1]
        h=(1-u)*(1-v)*z00+u*(1-v)*z10+(1-u)*v*z01+u*v*z11
        h=np.where(valid,h,np.nan)
        if not gradient:return h
        hx=((1-v)*(z10-z00)+v*(z11-z01))/self.dx
        hy=((1-u)*(z01-z00)+u*(z11-z10))/self.dy
        return h,np.where(valid,hx,np.nan),np.where(valid,hy,np.nan)
    def obs(self,X):return X[2]-self.evaluate(X[0],X[1])


def nominal(x=150000.,y=150000.):
    return np.array([x,y,8000.,V/np.sqrt(2),V/np.sqrt(2),0.])


def prior_sample(center,n,law,rng):
    if law=='gauss':u=rng.standard_normal((6,n))
    elif law=='uniforme':u=rng.uniform(-np.sqrt(3),np.sqrt(3),(6,n))
    else:raise ValueError('Loi inconnue')
    return np.asarray(center)[:,None]+SIGMA0[:,None]*u


def simulate(terrain,initial,rng,qscale=0.,steps=T,command=None):
    if command is None: command=np.zeros((6,steps+1))
    truth=np.empty((6,steps+1));truth[:,0]=initial
    for k in range(1,steps+1):
        truth[:,k]=F@truth[:,k-1]+command[:,k]+qscale*QSTD*rng.standard_normal(6)
    measures=terrain.obs(truth)+np.sqrt(R)*rng.standard_normal(steps+1)
    if not np.isfinite(measures).all():raise ValueError('Trajectoire sur un bord ou un trou du MNT')
    measures[0]=np.nan # pas de mesure initiale ; convention Y_1:k du cours
    return truth,measures


def psd_sqrt(P):
    """Racine dans des coordonnées adimensionnées, sans bruit artificiel ajouté."""
    scale=np.sqrt(np.maximum(np.diag(P),1e-24))
    A=P/scale[:,None]/scale[None,:]
    eig,U=np.linalg.eigh((A+A.T)/2)
    return scale[:,None]*(U*np.sqrt(np.maximum(eig,0))[None,:])


def kde_map(X,w,mean,P,previous=None):
    s=np.sqrt(np.maximum(np.diag(P),1e-20)); A=P/s[:,None]/s[None,:]
    eig,U=np.linalg.eigh((A+A.T)/2)
    L=s[:,None]*(U*np.sqrt(np.maximum(eig,1e-10))[None,:])
    Z=np.linalg.solve(L,X-mean[:,None]).T
    h=(4/(len(w)*8))**.1
    ids=np.searchsorted(np.cumsum(w),np.linspace(.05,.95,7));ids=np.minimum(ids,len(w)-1)
    C=np.vstack([np.zeros(6),Z[ids]])
    if previous is not None:C=np.vstack([C,np.linalg.solve(L,previous-mean)])
    lw=np.log(np.maximum(w,1e-300))
    for _ in range(15):
        a=-cdist(C,Z,'sqeuclidean')/(2*h*h)+lw
        a-=a.max(axis=1,keepdims=True);a=np.exp(a);a/=a.sum(axis=1,keepdims=True)
        new=a@Z
        if np.max(np.linalg.norm(new-C,axis=1))<1e-4:C=new;break
        C=new
    a=-cdist(C,Z,'sqeuclidean')/(2*h*h)+lw
    mx=a.max(axis=1);score=mx+np.log(np.exp(a-mx[:,None]).sum(axis=1))
    return mean+L@C[np.argmax(score)]


def bayesian_bound(terrain,center,draws=20000,seed=9173,steps=T):
    """BCR bayésienne pour X0~N(center,P0), Xk=F^k X0 EXACT (Q=0).
    J0=P0^-1; Jk=Jk-1+E[Gk.T Gk]/R, Gk=H(F^k X0)F^k.
    Calcul en coordonnées normalisées theta=D^-1(X0-center).
    Intégrale numérique sur des tirages gaussiens dans la zone intérieure.
    Un tirage hors domaine arrête le calcul (aucun gradient inventé).
    """
    rng=np.random.default_rng(seed);X=prior_sample(center,draws,'gauss',rng)
    J=np.eye(6);bound=np.empty((steps+1,6,6));bound[0]=np.diag(SIGMA0**2)
    outside=0
    for k in range(1,steps+1):
        X[:3]+=DT*X[3:]
        _,hx,hy=terrain.evaluate(X[0],X[1],True)
        valid=np.isfinite(hx)&np.isfinite(hy);outside+=np.sum(~valid)
        if not valid.all():raise ValueError('Tirages BCR hors MNT : choisir une zone intérieure')
        H=np.column_stack([-hx,-hy,np.ones(draws)])
        G=np.column_stack([H,H*k*DT])*SIGMA0
        J+=G.T@G/(draws*R)
        A=np.eye(6);A[:3,3:]=np.eye(3)*k*DT;A*=SIGMA0
        bound[k]=A@np.linalg.solve(J,A.T)
    return bound,outside


def loop_command(steps=T):
    """Virage planifié connu, sans accès à la position vraie.
    Période 720 s, rayon V*T/(2*pi). La commande vaut c_k-F*c_{k-1}.
    c est un plan relatif à l'origine, fixé avant les observations.
    """
    t=np.arange(steps+1)*DT;omega=2*np.pi/(T*DT)
    theta=np.pi/4+omega*t;r=V/omega
    c=np.zeros((6,steps+1))
    c[0]=r*(np.sin(theta)-np.sin(theta[0]))
    c[1]=-r*(np.cos(theta)-np.cos(theta[0]))
    c[3]=V*np.cos(theta);c[4]=V*np.sin(theta)
    u=np.zeros_like(c);u[:,1:]=c[:,1:]-F@c[:,:-1]
    return u
