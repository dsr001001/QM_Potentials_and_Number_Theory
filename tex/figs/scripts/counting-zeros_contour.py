import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

T = 40.0
tz = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([-1, 2, 2, -1, -1], [0, 0, T, T, 0], color="gray", lw=1.2)
ax.plot([2, 2, 0.5], [0, T, T], color="C0", lw=3, label=r"path $P$")
ax.axvline(0.5, color="k", ls="--", lw=0.8)
ax.plot([0.5] * len(tz), tz, "ro", ms=5, label="zeros below $T$")
ax.annotate("", xy=(2, 20), xytext=(2, 15), arrowprops=dict(arrowstyle="->", color="C0", lw=2))
ax.annotate("", xy=(1.0, T), xytext=(1.5, T), arrowprops=dict(arrowstyle="->", color="C0", lw=2))
for (x, y, s) in [(-1, 0, "$-1$"), (2, 0, "$2$"), (2, T, "$2+iT$"), (-1, T, "$-1+iT$")]:
    ax.text(x + 0.05, y + 0.8, s, fontsize=9)
ax.set_xlim(-1.6, 2.8); ax.set_ylim(-3, T + 4)
ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
ax.set_ylabel(r"$t=\mathrm{Im}\,s$")
ax.legend(loc="center left", fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("counting-zeros_contour.pdf", bbox_inches="tight")
