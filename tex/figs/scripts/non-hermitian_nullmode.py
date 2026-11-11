import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L,N=np.pi/2,400; h=2*L/(N+1)
_,psi=eigh_tridiagonal(np.full(N,2/h**2),np.full(N-1,-1/h**2),select="i",select_range=(0,59))
psi/=np.sqrt(h)
O64=psi.T@psi*h
p32=psi.astype(np.float32); O32=(p32.T@p32)*np.float32(h)
m=np.array([k for k in range(60) if k!=9])
y64=np.maximum(np.abs(O64[9,m]),1e-20); y32=np.maximum(np.abs(O32[9,m].astype(float)),1e-20)
fig,ax=plt.subplots(figsize=(6,4))
ax.semilogy(m+1,y32,'s',label='float32 eigenvectors')
ax.semilogy(m+1,y64,'o',label='float64 eigenvectors')
ax.axhline(1.2e-7,color='gray',ls='--',label='float32 epsilon')
ax.set_xlabel("partner level m")
ax.set_ylabel(r"$|(\phi_0)_{10,m}|$ (dimensionless)")
ax.legend()
plt.savefig("non-hermitian_nullmode.pdf", bbox_inches="tight")
