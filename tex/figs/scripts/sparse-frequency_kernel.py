import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dw = 8/450
j = np.arange(1, 451)
def D(tau):
    tau = np.asarray(tau, dtype=float)
    out = np.empty_like(tau)
    for i in range(0, tau.size, 2000):
        out[i:i+2000] = np.cos(np.outer(tau[i:i+2000], j*dw)).sum(axis=1)
    return out
Trep = 2*np.pi/dw
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ta = np.linspace(-2, 2, 2001)
ax[0].plot(ta, D(ta), lw=0.8)
ax[0].axhline(0, color="0.7", lw=0.5)
ax[0].set_xlabel(r"$\tau$ (height difference, dimensionless)")
ax[0].set_ylabel(r"$D(\tau)$")
ax[0].set_title(r"(a) near $\tau=0$", fontsize=9)
tb = np.linspace(-20, 400, 42001)
ax[1].plot(tb, D(tb), lw=0.3)
ax[1].axvline(Trep, ls=":", color="gray")
ax[1].text(Trep-5, 380, r"$T_{\rm rep}\approx353.4$", ha="right", fontsize=8)
ax[1].set_xlabel(r"$\tau$ (height difference, dimensionless)")
ax[1].set_ylabel(r"$D(\tau)$")
ax[1].set_title("(b) long range", fontsize=9)
plt.tight_layout()
plt.savefig("sparse-frequency_kernel.pdf", bbox_inches="tight")
