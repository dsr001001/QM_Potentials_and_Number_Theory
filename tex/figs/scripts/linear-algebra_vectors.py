import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
# left: addition
ax = axs[0]
u = np.array([3, 1]); v = np.array([1, 2]); w = u + v
for vec, col, lab in [(u, "C0", "u"), (v, "C1", "v"), (w, "C2", "u+v")]:
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.text(vec[0] * 1.02 + 0.05, vec[1] * 1.02 + 0.05, lab, color=col)
ax.plot([u[0], w[0]], [u[1], w[1]], "k--", lw=0.8)
ax.plot([v[0], w[0]], [v[1], w[1]], "k--", lw=0.8)
ax.set_xlim(-0.3, 4.8); ax.set_ylim(-0.3, 3.6); ax.set_aspect("equal")
ax.set_xlabel("first component (dimensionless)"); ax.set_ylabel("second component")
ax.set_title("addition")
# right: projection
ax = axs[1]
a = np.array([3, 1]); b = np.array([1, 2]); p = (a @ b) / (a @ a) * a
ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C0", lw=2))
ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C1", lw=2))
ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C3", lw=2))
ax.annotate("", xy=b, xytext=p, arrowprops=dict(arrowstyle="->", color="0.4", lw=1.5, ls="--"))
ax.text(3.05, 1.05, "a", color="C0"); ax.text(1.05, 2.05, "b", color="C1")
ax.text(1.5, 0.2, "proj = (1.5, 0.5)", color="C3", fontsize=8)
ax.text(0.2, 1.3, "residual", color="0.4", fontsize=8)
ax.set_xlim(-0.3, 3.6); ax.set_ylim(-0.3, 2.6); ax.set_aspect("equal")
ax.set_xlabel("first component (dimensionless)")
ax.set_title("projection")
plt.tight_layout()
plt.savefig("linear-algebra_vectors.pdf", bbox_inches="tight")
