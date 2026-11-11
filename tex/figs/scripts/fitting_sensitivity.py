import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import qmnt
L = 44.883; N = 3000; h = 2*L/(N+1); x = -L + h*np.arange(1, N+1)
V0 = 3.10073*np.pi
Vws = np.array([float(qmnt.wu_sprung_V(xi)) for xi in x])
ks = np.arange(1, 500)
C = np.cos(np.outer(x, ks)*np.pi/L)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for ax, V, vmin, title in [(axs[0], np.zeros(N), 0.0, "infinite well"), (axs[1], Vws, V0, "Wu-Sprung")]:
    E, v = eigh_tridiagonal(2/h**2 + V, -np.ones(N-1)/h**2, select='i', select_range=(0, 99))
    v = v/np.sqrt((v**2).sum(axis=0))
    A = (v**2).T @ C
    img = np.log10(np.maximum(np.abs(A), 1e-4))
    im = ax.imshow(img, origin='lower', aspect='auto', extent=[0.5, 499.5, 0.5, 100.5], vmin=-4, vmax=0, cmap='viridis')
    n = np.arange(1, 101)
    kc = 2*L*np.sqrt(np.maximum(E - vmin, 0))/np.pi
    ax.plot(kc, n, 'w--', lw=1)
    ax.set_xlim(1, 499); ax.set_ylim(0.5, 100.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("cosine index k (dimensionless)")
axs[0].set_ylabel("level index n")
fig.colorbar(im, ax=list(axs), label="log10 |A_nk|")
plt.savefig("fitting_sensitivity.pdf", bbox_inches="tight")
