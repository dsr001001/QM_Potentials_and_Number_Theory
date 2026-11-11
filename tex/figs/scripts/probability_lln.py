import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

nmax = 10000
k = np.arange(1, nmax + 1)
fig, ax = plt.subplots(figsize=(6, 4))
for seed, style in zip([1, 2, 3], ["b-", "r-", "g-"]):
    x = np.random.default_rng(seed).exponential(1.0, nmax)
    ax.plot(k, np.cumsum(x) / k, style, lw=1, label=f"seed {seed}")
ax.plot(k, 1 + 1 / np.sqrt(k), "k--", lw=1, label=r"$1\pm1/\sqrt{n}$")
ax.plot(k, 1 - 1 / np.sqrt(k), "k--", lw=1)
ax.axhline(1.0, color="k", lw=0.5)
ax.set_xscale("log")
ax.set_ylim(0, 2)
ax.set_xlabel("number of draws $n$")
ax.set_ylabel(r"running mean $\bar x_n$ (units of the true mean)")
ax.legend()
plt.savefig("probability_lln.pdf", bbox_inches="tight")
