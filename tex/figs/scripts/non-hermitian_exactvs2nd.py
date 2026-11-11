import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eig
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
K=60; nn=np.arange(1,K+1); E=nn**2.0
S=np.array([[s1(a,b) for b in nn] for a in nn])
dE=E[:,None]-E[None,:]; np.fill_diagonal(dE,np.inf); T=(S**2/dE).sum(1)
def exact(c):
    l,v=eig(np.diag(E)+1j*c*S); mt=np.argmax(np.abs(v)**2,0); ex=np.full(K,np.nan)
    for j in range(K): ex[mt[j]]=l[j].real-E[mt[j]]
    return ex
fig,axs=plt.subplots(1,2,figsize=(6.5,3))
for ax,c in zip(axs,(0.5,4.0)):
    ex=exact(c)[:50]; so=-c**2*T[:50]
    ax.plot(so,ex,'o',ms=3)
    lo=np.nanmin([so.min(),np.nanmin(ex)]); hi=np.nanmax([so.max(),np.nanmax(ex)])
    ax.plot([lo,hi],[lo,hi],'k--',lw=0.8)
    ax.set_xlabel('second-order shift (energy units)'); ax.set_ylabel('exact real shift (energy units)'); ax.set_title(f'c={c}')
plt.tight_layout()
plt.savefig("non-hermitian_exactvs2nd.pdf", bbox_inches="tight")
