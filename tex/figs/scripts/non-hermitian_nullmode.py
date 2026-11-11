import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L,N=np.pi/2,400; h=2*L/(N+1)
E,psi=eigh_tridiagonal(np.full(N,2/h**2),np.full(N-1,-1/h**2),select="i",select_range=(0,59))
psi/=np.sqrt(h)
o64=np.abs(psi.T@psi[:,9]*h)
p32=psi.astype(np.float32); o32=np.abs((p32.T@p32[:,9])*np.float32(h)).astype(float)
m=np.arange(1,61); k=m!=10
fig,ax=plt.subplots(figsize=(6,4))
ax.semilogy(m[k],np.maximum(o64[k],1e-20),"o",label="float64")
ax.semilogy(m[k],np.maximum(o32[k],1e-20),"s",label="float32")
ax.set_xlabel("partner level m"); ax.set_ylabel(r"$|(\phi_0)_{10,m}|$ (dimensionless)"); ax.legend()
plt.savefig("non-hermitian_nullmode.pdf",bbox_inches="tight")
