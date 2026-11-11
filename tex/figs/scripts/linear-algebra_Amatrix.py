import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scipy.linalg as sla
import qmnt

nlev, M = 30, 30
L = float(qmnt.L_of(1000))
# box, analytic
Abox = np.zeros((nlev, M))
Abox[:, 0] = 1.0
for n in range(1, nlev + 1):
    if n < M:
        Abox[n - 1, n] = (-1)**(n + 1) / 2
# Wu-Sprung, finite differences
N = 1200
h = 2 * L / (N + 1)
x = -L + h * np.arange(1, N + 1)
V = np.array([float(qmnt.wu_sprung_V(abs(xx))) for xx in x])
d = 2.0 / h**2 + V
e = -np.ones(N - 1) / h**2
w, vec = sla.eigh_tridiagonal(d, e, select="i", select_range=(0, nlev - 1))
psi = vec / np.sqrt(h)                     # sum psi^2 h = 1
Phi = np.cos(np.outer(x, np.arange(M)) * np.pi / L)
Aws = h * (psi**2).T @ Phi
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
for ax, A, t in zip(axs, [Abox, Aws], ["box", "Wu-Sprung"]):
    im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper",
                   extent=[-0.5, M - 0.5, nlev + 0.5, 0.5], aspect="auto")
    ax.set_xlabel("basis index k"); ax.set_title(t)
axs[0].set_ylabel("level n")
fig.colorbar(im, ax=axs, shrink=0.8, label="A_nk (dimensionless)")
plt.savefig("linear-algebra_Amatrix.pdf", bbox_inches="tight")
