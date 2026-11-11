import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

rng = np.random.default_rng(1)
pts = np.sort(rng.uniform(0.0, 10000.0, 10000))
gaps = np.diff(pts)

t = qmnt.zeros(10000)
u = qmnt.Nsm(t)
sp = np.diff(u)

s = np.linspace(0, 4, 400)
bins = np.linspace(0, 4, 41)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
axs[0].hist(gaps, bins=bins, density=True, color="0.75", edgecolor="k")
axs[0].plot(s, np.exp(-s), "r-", lw=2, label=r"$e^{-s}$")
axs[0].set_title("independent points (Poisson)")
axs[1].hist(sp, bins=bins, density=True, color="0.75", edgecolor="k")
axs[1].plot(s, np.exp(-s), "r-", lw=2, label=r"$e^{-s}$")
axs[1].set_title(r"zeta zeros, first $10^4$")
for ax in axs:
    ax.set_xlabel("spacing $s$ (mean spacings)")
    ax.legend()
axs[0].set_ylabel("density (per mean spacing)")
plt.tight_layout()
plt.savefig("probability_exponential.pdf", bbox_inches="tight")
