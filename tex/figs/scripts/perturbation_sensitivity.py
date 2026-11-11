import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import qmnt
L,N,nl,M=44.883,3000,100,100
h=2*L/(N+1); x=-L+h*np.arange(1,N+1)
phi=np.cos(np.outer(x,np.arange(M))*np.pi/L)
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
for a,V,t in zip(ax,[np.zeros(N),np.array([qmnt.wu_sprung_V(xx) for xx in x])],["box","Wu--Sprung"]):
    E,p=eigh_tridiagonal(2/h**2+V,-np.ones(N-1)/h**2,select='i',select_range=(0,nl-1))
    p=p/np.sqrt(h); A=(p**2*h).T@phi
    im=a.imshow(np.log10(np.abs(A)+1e-4),origin="lower",aspect="auto",extent=[0,M,1,nl],vmin=-4,vmax=0)
    a.set_title(t); a.set_xlabel("k"); a.set_ylabel("n")
fig.colorbar(im,ax=ax,label=r"$\log_{10}|A_{nk}|$")
plt.savefig("perturbation_sensitivity.pdf",bbox_inches="tight")
