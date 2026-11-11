import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import qmnt
L = 44.883; N = 3000
x = qmnt.grid(L, N); h = 2*L/(N+1)
ks = np.arange(1, 500)
phi = np.cos(np.outer(ks, np.pi*x/L))
def Amat(V):
    d = 2/h**2 + V; e = -np.ones(N-1)/h**2
    E, v = eigh_tridiagonal(d, e, select='i', select_range=(0, 99))
    return E, (v**2).T @ phi.T
Vws = np.array([qmnt.wu_sprung_V(xi) for xi in x])
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
for ax, V, vmin, title in [(axs[0], np.zeros(N), 0.0, "box"),
                           (axs[1], Vws, qmnt.V0, "Wu--Sprung")]:
    E, A = Amat(V)
    im = ax.imshow(np.log10(np.abs(A) + 1e-4), origin="lower", aspect="auto",
                   extent=[1, 499, 1, 100], vmin=-4, vmax=0, cmap="viridis")
    n = np.arange(1, 101)
    ax.plot(2*L*np.sqrt(np.maximum(E - vmin, 0))/np.pi, n, "w--", lw=1)
    ax.set_xlim(1, 499); ax.set_ylim(1, 100)
    ax.set_xlabel("cosine index k"); ax.set_title(title)
axs[0].set_ylabel("level n")
fig.colorbar(im, ax=axs, label=r"$\log_{10}|A_{nk}|$")
plt.savefig("fitting_sensitivity.pdf", bbox_inches="tight")
