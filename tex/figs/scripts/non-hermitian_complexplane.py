import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eig
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
def c1(a,b): return -0.5*(int(abs(a-b)==2)-int(a+b==2))
K=60; nn=np.arange(1,K+1); E=nn**2.0
S=np.array([[s1(a,b) for b in nn] for a in nn])
C=np.eye(K)+np.array([[c1(a,b) for b in nn] for a in nn])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3))
for c,mk in ((1,'o'),(2.5,'s'),(4,'^')):
    l=eig(np.diag(E)+1j*c*S,right=False)
    ax1.plot(l.real,l.imag,mk,ms=4,mfc='none',label=f'c={c}')
ax1.set_xlim(-2,70); ax1.set_xlabel(r'Re $\lambda$ (energy units)'); ax1.set_ylabel(r'Im $\lambda$ (energy units)'); ax1.set_title('PT-odd'); ax1.legend(fontsize=8)
for g,mk in ((0.5,'o'),(2,'s')):
    l=eig(np.diag(E)-1j*g*C,right=False)
    ax2.plot(l.real,l.imag,mk,ms=4,mfc='none',label=f'g={g}')
ax2.axhline(0,color='gray',lw=0.5)
ax2.set_xlim(-2,70); ax2.set_xlabel(r'Re $\lambda$ (energy units)'); ax2.set_ylabel(r'Im $\lambda$ (energy units)'); ax2.set_title('absorptive-even'); ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("non-hermitian_complexplane.pdf", bbox_inches="tight")
