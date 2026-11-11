import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 40, 4000)
L2 = np.log(2.0)
def wave(t, mmax):
    out = np.zeros_like(t)
    for m in range(1, mmax + 1):
        out += np.sin(m * t * L2) / (m * 2**(m / 2))
    return -out / np.pi
w1, w3, w40 = wave(t, 1), wave(t, 3), wave(t, 40)
bound = 0.048698
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
ax = axs[0]
ax.plot(t, w1, lw=0.8, label=r"$m=1$")
ax.plot(t, w3, lw=1.2, label=r"$m\leq3$")
ax.plot(t, w40, "k--", lw=0.8, label=r"$m\leq40$")
ax.set_xlabel(r"height $t$ (dimensionless)")
ax.set_ylabel(r"wave (number of zeros)")
ax.legend(fontsize=7)
ax = axs[1]
ax.plot(t, w40 - w3, lw=1)
ax.axhline(bound, color="r", ls="--", lw=0.8)
ax.axhline(-bound, color="r", ls="--", lw=0.8)
ax.set_xlabel(r"height $t$ (dimensionless)")
ax.set_ylabel(r"difference (number of zeros)")
ax.set_title(r"$m\leq40$ minus $m\leq3$, bound $\pm0.0487$", fontsize=8)
plt.tight_layout()
plt.savefig("series_re2.pdf", bbox_inches="tight")
