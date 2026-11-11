import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L, N = np.pi/2, 800; h = 2*L/(N+1)
E, psi = eigh_tridiagonal(2/h**2*np.ones(N), -1/h**2*np.ones(N-1), select="i", select_range=(0,59))
psi /= np.sqrt((psi**2).sum(axis=0)*h)
G64 = psi.T@psi*h
p32 = psi.astype(np.float32); G32 = (p32.T@p32)*np.float32(h)
off = ~np.eye(60, dtype=bool)
a64 = np.log10(abs(G64[off])+1e-30); a32 = np.log10(abs(G32[off]).astype(float)+1e-30)
plt.figure(figsize=(6,4))
plt.hist(a64[a64>-25], bins=40, alpha=0.6, label="float64")
plt.hist(a32[a32>-25], bins=40, alpha=0.6, label="float32")
plt.xlabel(r"$\log_{10}|\langle n|m\rangle|$, $n\neq m$ (dimensionless)"); plt.ylabel("count")
plt.legend()
plt.savefig("paper-spurious-floor_float32.pdf", bbox_inches="tight")
