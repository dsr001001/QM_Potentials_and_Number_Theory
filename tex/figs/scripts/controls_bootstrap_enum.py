import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from itertools import product

x = np.array([1.0, 2.0, 6.0])
counts = {}
for idx in product(range(3), repeat=3):
    m = round(float(x[list(idx)].mean()), 10)
    counts[m] = counts.get(m, 0) + 1
vals = np.array(sorted(counts))
w = np.array([counts[v] for v in vals])
prob = w / 27.0

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(4/3, 14/3, color="orange", alpha=0.15, label="80% percentile interval")
ax.bar(vals, prob, width=0.12, color="steelblue", edgecolor="k")
for v, p, k in zip(vals, prob, w):
    ax.text(v, p + 0.006, f"{k}/27", ha="center", fontsize=8)
ax.axvline(3.0, color="k", ls="--", lw=1, label="sample mean = 3")
ax.set_xlabel("bootstrap mean (dimensionless data units)")
ax.set_ylabel("probability")
ax.set_ylim(0, 0.27)
ax.legend(loc="upper right", fontsize=8)
plt.savefig("controls_bootstrap_enum.pdf", bbox_inches="tight")
