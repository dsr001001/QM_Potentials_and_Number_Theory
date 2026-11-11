import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

n, ell = 12, 3
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
for ax, circ in zip(axes, [False, True]):
    nbl = n if circ else n - ell + 1
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color="0.7"))
        ax.text(i + 0.45, 0.45, str(i + 1), ha="center", va="center", fontsize=7)
    for j in range(nbl):
        y = -1 - j
        for a in range(ell):
            c = j + a
            wrap = c >= n
            ax.add_patch(plt.Rectangle((c % n, y), 0.9, 0.8,
                         color="orange" if wrap else "steelblue"))
    ax.set_xlim(-0.5, n + 0.5); ax.set_ylim(-nbl - 1.5, 1.5)
    ax.set_xlabel("observation index")
    ax.set_yticks([])
    ax.set_title("circular block bootstrap" if circ else "moving block bootstrap", fontsize=9)
plt.tight_layout()
plt.savefig("controls_blocks.pdf", bbox_inches="tight")
