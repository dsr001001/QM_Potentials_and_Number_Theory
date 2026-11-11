import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(-1, 0.97, 500)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, -np.log(1 - x), "k", lw=2.5, label=r"$-\log(1-x)$")
for n in [1, 2, 3, 5, 10]:
    m = np.arange(1, n + 1)
    y = np.array([np.sum(xx**m / m) for xx in x])
    ax.plot(x, y, lw=1, label=rf"$n={n}$")
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel(r"$x$ (dimensionless)")
ax.set_ylabel(r"value (dimensionless)")
ax.set_ylim(-1.0, 3.5)
ax.legend(fontsize=8, loc="upper left")
plt.savefig("series_logseries.pdf", bbox_inches="tight")
