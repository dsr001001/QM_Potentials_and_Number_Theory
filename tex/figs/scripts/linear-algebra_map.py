import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = np.array([[2, 1], [1, 2]])
sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
pg = A @ sq
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill(sq[0], sq[1], alpha=0.3, color="C0", label="unit square, area 1")
ax.fill(pg[0], pg[1], alpha=0.3, color="C1", label="image, area det = 3")
ax.plot(sq[0], sq[1], "C0"); ax.plot(pg[0], pg[1], "C1")
ax.set_aspect("equal"); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
ax.set_xlabel("x (dimensionless)"); ax.set_ylabel("y (dimensionless)")
ax.legend()
plt.savefig("linear-algebra_map.pdf", bbox_inches="tight")
