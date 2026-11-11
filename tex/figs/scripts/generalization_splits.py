import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

n = np.arange(1, 51)
rng = np.random.default_rng(1)
test_f = n > 40
test_r = np.zeros(50, bool); test_r[rng.choice(50, 10, replace=False)] = True
test_o = (n % 2 == 0)
test_g = (n >= 21) & (n <= 30)
labels = ["forward in height", "random 80/20", "odd/even", "interpolation gap"]

fig, ax = plt.subplots(figsize=(6, 4))
for row, test in enumerate([test_f, test_r, test_o, test_g]):
    yv = 3 - row
    ax.plot(n[~test], np.full((~test).sum(), yv), "o", color="C0", ms=5)
    ax.plot(n[test], np.full(test.sum(), yv), "o", mfc="white", mec="C3", ms=5)
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(labels)
ax.set_ylim(-0.7, 3.7)
ax.set_xlabel("level index $n$ (schematic, 50 levels; dimensionless)")
handles = [Line2D([], [], marker="o", color="C0", ls="", label="training"),
           Line2D([], [], marker="o", mfc="white", mec="C3", ls="", label="held out")]
ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.12))
plt.savefig("generalization_splits.pdf", bbox_inches="tight")
