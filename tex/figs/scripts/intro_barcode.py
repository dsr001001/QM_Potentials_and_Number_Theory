import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

tz = np.asarray(qmnt.zeros(29))
rng = np.random.default_rng(1)
rnd = np.sort(rng.uniform(tz[0], tz[-1], 29))
eq = np.linspace(tz[0], tz[-1], 29)

fig, ax = plt.subplots(figsize=(6, 4))
rows = [(2, tz, "zeros $t_n$", "k"), (1, rnd, "random points", "tab:orange"),
        (0, eq, "equally spaced", "tab:blue")]
for y, pts, lab, col in rows:
    ax.vlines(pts, y - 0.35, y + 0.35, color=col, lw=1.5)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels([r[2] for r in rows])
ax.set_xlabel(r"height $t$ (dimensionless)")
ax.set_xlim(10, 102)
ax.set_ylim(-0.6, 2.6)
ax.grid(axis="x", alpha=0.3)
plt.savefig("intro_barcode.pdf", bbox_inches="tight")
