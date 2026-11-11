import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = 450; dw = 8/450
j = np.arange(1, J+1)*dw
def K(tau):
    out = np.empty(len(tau))
    for i in range(0, len(tau), 2000):
        out[i:i+2000] = np.cos(np.outer(tau[i:i+2000], j)).sum(axis=1)
    return out
P = 2*np.pi/dw
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ta = np.linspace(-3, 3, 3001)
ax[0].plot(ta, K(ta), lw=0.8)
ax[0].set_xlabel(r"$\tau$ (height units, dimensionless)")
ax[0].set_ylabel(r"$K(\tau)$")
ax[0].set_title(r"(a) near $\tau=0$", fontsize=9)
tb = np.linspace(0, 800, 40001)
ax[1].plot(tb, K(tb), lw=0.4)
for k in (1, 2):
    ax[1].axvline(k*P, ls=":", color="gray")
ax[1].text(P, 470, r"$P$", ha="center", fontsize=8)
ax[1].text(2*P, 470, r"$2P$", ha="center", fontsize=8)
ax[1].set_ylim(-60, 500)
ax[1].set_xlabel(r"$\tau$ (height units, dimensionless)")
ax[1].set_title("(b) long range", fontsize=9)
plt.tight_layout()
plt.savefig("sparse-frequency_kernel.pdf", bbox_inches="tight")
