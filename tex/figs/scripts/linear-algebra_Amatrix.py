import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
L = 44.883; nlev = 30; M = 30
n = np.arange(1, nlev+1); k = np.arange(M)
Abox = np.zeros((nlev, M)); Abox[:, 0] = 1
for nn in range(1, nlev+1):
    if nn < M:
        Abox[nn-1, nn] = (-1)**(nn+1)/2
N = 1200
x = qmnt.grid(L, N); h = 2*L/(N+1)
V = np.array([qmnt.wu_sprung_V(xi) for xi in x])
E, vecs = qmnt.fd_levels(V, L, k=nlev, return_vectors=True)
vecs = np.asarray(vecs)
if vecs.shape[0] != N:
    vecs = vecs.T
vecs = vecs[:, :nlev]
vecs = vecs/np.sqrt(h*np.sum(vecs**2, axis=0))
Phi = np.cos(np.outer(x, k)*np.pi/L)
Aws = h*(vecs**2).T @ Phi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
for ax, Am, ttl in [(ax1, Abox, "infinite well"), (ax2, Aws, "Wu-Sprung")]:
    im = ax.imshow(Am, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper",
                   extent=[-0.5, M-0.5, nlev+0.5, 0.5])
    ax.set_xlabel("basis index k"); ax.set_title(ttl)
ax1.set_ylabel("level n")
fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="A_nk (dimensionless)")
plt.savefig("linear-algebra_Amatrix.pdf", bbox_inches="tight")
