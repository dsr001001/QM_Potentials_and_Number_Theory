import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = np.array([1.0, 0.0]); b = np.array([0.5, np.sqrt(3)/2])
y = a + 0.8*b
fig, ax = plt.subplots(figsize=(6, 4))
def arrow(p, q, **kw):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", **kw))
arrow((0, 0), a, color="tab:blue"); ax.text(1.0, -0.07, r"$\mathbf{a}$", color="tab:blue")
arrow((0, 0), b, color="tab:green"); ax.text(0.42, 0.9, r"$\mathbf{b}$", color="tab:green")
arrow((0, 0), y, color="k", lw=1.5); ax.text(y[0]+0.03, y[1]+0.03, r"$\mathbf{y}$")
p1 = 1.4*a
arrow((0, 0), p1, color="tab:orange", ls="--")
end = p1 + 0.6*b
arrow(p1, end, color="tab:orange", ls="--")
ax.plot(*end, "o", mfc="white", mec="tab:orange")
ax.text(end[0]+0.03, end[1]-0.08, "plain pursuit", color="tab:orange", fontsize=8)
ax.plot(*y, "ko")
ax.text(y[0]-0.45, y[1]+0.08, "orthogonal pursuit", fontsize=8)
ax.set_xlim(-0.1, 2.0); ax.set_ylim(-0.15, 1.1); ax.set_aspect("equal")
ax.set_xlabel("first component (dimensionless)")
ax.set_ylabel("second component (dimensionless)")
plt.savefig("sparse-frequency_pursuit.pdf", bbox_inches="tight")
