import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

x = np.linspace(-1.9, 0.9, 700)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(-1.9, -1.0, color="0.9", label="series diverges ($x\\leq -1$)")
ax.plot(x, 1.0 / (1.0 - x), "k", lw=2.5, label=r"$1/(1-x)$")
for N, ls in [(1, ":"), (2, "--"), (5, "-."), (20, "-")]:
    S = np.zeros_like(x)
    for k in range(N + 1):
        S += x**k
    ax.plot(x, S, ls, lw=1.2, label=r"$S_{%d}(x)$" % N)
ax.axhline(0, color="0.5", lw=0.5)
ax.set_ylim(-3, 8)
ax.set_xlim(-1.9, 0.9)
ax.set_xlabel(r"$x$ (dimensionless)")
ax.set_ylabel(r"value (dimensionless)")
ax.legend(fontsize=8, loc="upper left")
plt.savefig("zeta_geom.pdf", bbox_inches="tight")
