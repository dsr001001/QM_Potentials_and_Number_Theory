import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eig
K=60; n=np.arange(1,K+1); E=n**2.0
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
S=np.array([[s1(a,b) for b in n] for a in n])
C=np.array([[ (1.0 if a==b else 0)+(0.5*(-1) if False else 0) for b in n] for a in n])
def c1(a,b):  # (cos_1)_{ab}
    return -0.5*(int(abs(a-b)==2)-int(a+b==2))
Cos=np.eye(K)+np.array([[c1(a,b) for b in n] for a in n])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3))
for c,mk in [(1,"o"),(2.5,"s"),(4,"^")]:
    l=eig(np.diag(E)+1j*c*S,right=False); ax1.plot(l.real,l.imag,mk,ms=4,label=f"c={c}")
for g,mk in [(0.5,"o"),(2,"s")]:
    l=eig(np.diag(E)-1j*g*Cos,right=False); ax2.plot(l.real,l.imag,mk,ms=4,label=f"g={g}")
for ax in (ax1,ax2):
    ax.set_xlim(0,120); ax.set_xlabel("Re eigenvalue (energy units)"); ax.set_ylabel("Im eigenvalue"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("non-hermitian_complexplane.pdf",bbox_inches="tight")
