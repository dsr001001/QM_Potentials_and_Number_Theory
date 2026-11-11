import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import qmnt

L, N = 44.883, 4000
h = 2*L/(N+1)
x = -L + h*np.arange(1, N+1)
V = np.array([qmnt.wu_sprung_V(xi) for xi in x])
d = 2/h**2 + V; e = -np.ones(N-1)/h**2
E, psi = eigh_tridiagonal(d, e, select='i', select_range=(0, 19))
psi = psi/np.sqrt(h)

fig, ax = plt.subplots(figsize=(6, 4))
m = np.abs(x) <= 8
ax.plot(x[m], V[m], "k-", lw=1.5, label="$V(x)$")
for n in [1, 2, 3, 4, 20]:
    y = psi[:, n-1]
    ax.plot(x[m], E[n-1] + 4*y[m], lw=1)
    ax.axhline(E[n-1], color="gray", lw=0.4, ls=":")
    ax.text(8.1, E[n-1], f"$n={n}$", fontsize=8, va="center")
ax.set_xlim(-8, 9)
ax.set_ylim(V.min()-2, E[19]+8)
ax.set_xlabel("$x$ (units $\\hbar=2m=1$)")
ax.set_ylabel("energy (units $\\hbar=2m=1$)")
ax.legend(fontsize=8, loc="upper left")
plt.savefig("eigen_wavefunctions.pdf", bbox_inches="tight")
