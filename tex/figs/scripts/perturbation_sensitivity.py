import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.linalg import eigh_tridiagonal
import qmnt
L=44.883; N=3000; h=2*L/(N+1); x=-L+h*np.arange(1,N+1)
k=np.arange(100); phi=np.cos(np.outer(x,k)*np.pi/L)
def Amat(V):
    E,p=eigh_tridiagonal(2/h**2+V,-np.ones(N-1)/h**2,select='i',select_range=(0,99))
    p=p/np.sqrt(h); return (p**2*h).T@phi
Abox=Amat(np.zeros(N))
Vws=np.array([qmnt.wu_sprung_V(xx) for xx in x])
Aws=Amat(Vws)
fig,axs=plt.subplots(1,2,figsize=(6.5,3))
for ax,A,t in zip(axs,[Abox,Aws],['box','Wu--Sprung']):
    im=ax.imshow(np.abs(A)+1e-12,origin='lower',aspect='auto',norm=LogNorm(vmin=1e-4,vmax=1),extent=[-0.5,99.5,0.5,100.5],cmap='viridis')
    ax.set_xlabel('basis index k'); ax.set_title(t)
axs[0].set_ylabel('level n')
fig.colorbar(im,ax=axs,label=r'$|A_{nk}|$ (dimensionless)')
plt.savefig("perturbation_sensitivity.pdf", bbox_inches="tight")
