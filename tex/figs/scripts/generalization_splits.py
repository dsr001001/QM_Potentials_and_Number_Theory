import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = np.arange(1, 51)
rng = np.random.default_rng(1)
rand_test = np.zeros(50, bool); rand_test[rng.choice(50, 10, replace=False)] = True
splits = [("forward", n > 40), ("random 80/20", rand_test), ("odd/even", n % 2 == 0), ("interpolation gap", (n > 20) & (n <= 30))]
fig, ax = plt.subplots(figsize=(6, 4))
for row, (name, test) in enumerate(splits):
    yv = 3 - row
    ax.plot(n[~test], np.full((~test).sum(), yv), "ko", ms=4)
    ax.plot(n[test], np.full(test.sum(), yv), "o", mfc="white", mec="k", ms=4)
    ax.text(0, yv + 0.3, name, fontsize=9)
ax.set_yticks([]); ax.set_ylim(-0.6, 3.8); ax.set_xlim(-1, 52)
ax.set_xlabel(r"level index $n$ (dimensionless)")
ax.plot([], [], "ko", label="training"); ax.plot([], [], "o", mfc="white", mec="k", label="held out")
ax.legend(fontsize=8, loc="lower right")
plt.savefig("generalization_splits.pdf", bbox_inches="tight")
