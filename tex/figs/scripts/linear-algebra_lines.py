import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

x = np.linspace(0, 4, 200)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for ax, three in zip(axs, [False, True]):
    ax.plot(x, 3 - x, label="x + y = 3")
    ax.plot(x, x - 1, label="x - y = 1")
    if three:
        ax.plot(x, (5 - x) / 2, label="x + 2y = 5")
        ax.set_title("3 equations, 2 unknowns")
    else:
        ax.plot([2], [1], "ko")
        ax.text(2.1, 1.15, "(2, 1)")
        ax.set_title("2 equations, 2 unknowns")
    ax.set_xlim(0, 4); ax.set_ylim(-1, 3.5)
    ax.set_xlabel("x (dimensionless)")
    ax.legend(fontsize=7, loc="upper right")
axs[0].set_ylabel("y (dimensionless)")
plt.tight_layout()
plt.savefig("linear-algebra_lines.pdf", bbox_inches="tight")
