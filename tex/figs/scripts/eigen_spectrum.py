import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

L, N = 44.883, 600
h = 2*L/(N+1)
d = 2/h**2*np.ones(N); e = -np.ones(N-1)/h**2
E = eigh_tridiagonal(d, e, eigvals_only=True)
k = np.arange(1, N+1)
exact = (k*np.pi/(2*L))**2

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(k, exact, "b-", lw=1, label="exact $(k\\pi/2L)^2$")
a1.plot(k[::10], E[::10], "r.", ms=3, label="matrix $D_N/h^2$")
a1.set_xlabel("level index $k$")
a1.set_ylabel("eigenvalue (units $\\hbar=2m=1$)")
a1.legend(fontsize=8)
a2.plot(k, E/exact, "k-")
a2.axhline(4/np.pi**2, color="gray", ls="--", lw=0.8)
a2.set_xlabel("level index $k$")
a2.set_ylabel("matrix / exact (dimensionless)")
plt.tight_layout()
plt.savefig("eigen_spectrum.pdf", bbox_inches="tight")
