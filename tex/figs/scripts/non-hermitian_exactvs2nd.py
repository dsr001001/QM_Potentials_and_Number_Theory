import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eig
K=60; n=np.arange(1,K+1); E=n**2.0
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
S=np.array([[s1(a,b) for b in n] for a in n])
dE=E[:,None]-E[None,:]; np.fill_diagonal(dE,np.inf)
T=(S**2/dE).sum(1)
fig,axs=plt.subplots(1,2,figsize=(6.5,3))
for ax,c in zip(axs,[0.5,4]):
    l,v=eig(np.diag(E)+1j*c*S); mt=np.argmax(np.abs(v)**2,0)
    ex=np.full(K,np.nan)
    for j in range(K): ex[mt[j]]=l[j].real-E[mt[j]]
    so=-c**2*T
    ax.plot(so[:50],ex[:50],"o",ms=3); lo=np.nanmin([so[:50].min(),np.nanmin(ex[:50])]); hi=np.nanmax([so[:50].max(),np.nanmax(ex[:50])])
    ax.plot([lo,hi],[lo,hi],"k--"); ax.set_title(f"c={c}")
    ax.set_xlabel("second-order shift"); ax.set_ylabel("exact real shift")
plt.tight_layout(); plt.savefig("non-hermitian_exactvs2nd.pdf",bbox_inches="tight")
