import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

N = 20
D = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
w, U = np.linalg.eigh(D)
j = np.arange(1, N+1)
s = j/(N+1)
sfine = np.linspace(0, 1, 500)

fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
for ax, k in zip(axes.ravel(), [1,2, 3, 20]):
    v = U[:, k-1]
    exact = np.sin(k*np.pi*s)
    # fix sign so that numerical vector matches the sine
    if np.dot(v, exact) < 0:
        v = -v
    v = v/np.max(np.abs(v))
    ax.plot(sfine, np.sin(k*np.pi*sfine), "b-", lw=1)
    ax.plot(s, v, "ro", ms=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"mode $k={k}$, $\\lambda={w[k-1]:.4f}$", fontsize=9)
    ax.set_ylim(-1.2, 1.2)
for ax in axes[1]:
    ax.set_xlabel("position $s=j/(N+1)$ (dimensionless)")
for ax in axes[:, 0]:
    ax.set_ylabel("component (scaled)")
plt.tight_layout()
plt.savefig("eigen_modes.pdf", bbox_inches="tight")
