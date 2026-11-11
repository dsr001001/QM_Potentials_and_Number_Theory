import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 44.883
x = np.linspace(-L, L, 2001)
fig, axs = plt.subplots(1, 3, figsize=(6.5, 3), sharey=True)
for k in range(4):
    axs[0].plot(x, np.cos(k*np.pi*x/L), label="k=%d" % k)
y = 2*np.abs(x)/L - 1
for k in range(4):
    axs[1].plot(x, np.cos(k*np.arccos(np.clip(y, -1, 1))), label="k=%d" % k)
M = 5; u = np.abs(x)/L
for j in range(M):
    axs[2].plot(x, np.maximum(0, 1 - (M-1)*np.abs(u - j/(M-1))), label="j=%d" % j)
for ax, t in zip(axs, ["cosine", "Chebyshev", "hat (M=5)"]):
    ax.set_title(t, fontsize=9); ax.set_xlabel("x (units hbar=2m=1)"); ax.legend(fontsize=6)
axs[0].set_ylabel("phi_k(x) (dimensionless)")
plt.tight_layout()
plt.savefig("fitting_bases.pdf", bbox_inches="tight")
