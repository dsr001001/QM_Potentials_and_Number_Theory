import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

M = np.array([[2, 1], [1, 2]])
sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
im = M @ sq
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill(sq[0], sq[1], color="C0", alpha=0.4, label="unit square, area 1")
ax.fill(im[0], im[1], color="C1", alpha=0.3, label="image, area det = 3")
ax.plot(im[0], im[1], "C1")
ax.annotate("", xy=(2, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k"))
ax.annotate("", xy=(1, 2), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k"))
ax.text(2.05, 0.9, "M e1 = (2, 1)"); ax.text(0.5, 2.1, "M e2 = (1, 2)")
ax.set_aspect("equal"); ax.set_xlim(-0.3, 3.5); ax.set_ylim(-0.3, 3.5)
ax.set_xlabel("first component (dimensionless)")
ax.set_ylabel("second component (dimensionless)")
ax.legend(loc="lower right", fontsize=8)
plt.savefig("linear-algebra_map.pdf", bbox_inches="tight")
