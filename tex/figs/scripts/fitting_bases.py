import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 44.883; x = np.linspace(-L, L, 1201); y = 2*np.abs(x)/L - 1; u = np.abs(x)/L
fig, axs = plt.subplots(1, 3, figsize=(6.5, 3), sharey=True)
for k in range(4):
    axs[0].plot(x, np.cos(k*np.pi*x/L), label="k=%d" % k)
    axs[1].plot(x, np.cos(k*np.arccos(np.clip(y, -1, 1))), label="k=%d" % k)
M = 5
for j in range(M):
    axs[2].plot(x, np.maximum(0, 1 - (M-1)*np.abs(u - j/(M-1))), label="j=%d" % j)
for ax, t in zip(axs, ["cosine", "Chebyshev", "hat, M=5"]):
    ax.set_title(t); ax.set_xlabel("x"); ax.legend(fontsize=5)
axs[0].set_ylabel(r"$\phi(x)$")
plt.tight_layout(); plt.savefig("fitting_bases.pdf", bbox_inches="tight")
