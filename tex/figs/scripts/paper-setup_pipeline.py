import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, txt, fc):
    ax.text(x, y, txt, ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec="k", lw=0.8))

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.0))

box(1.1, 4.2, "zeros $t_n$\n(mpmath, step 1)", "#dde6f5")
box(1.1, 2.5, "smooth part\n$N_{\\mathrm{sm}}$, $s_n$, gap$_n$\n(step 2)", "#dde6f5")
box(3.7, 4.2, "Wu-Sprung $V$, Eq. (2)\nFD solver (steps 3-4)", "#e6f2dd")
box(3.7, 2.5, "targets $\\tilde t_n = t_n + \\Delta_n$\n(step 5)", "#e6f2dd")
box(6.3, 4.2, "fit $\\delta V$ on zeros\n$1,\\ldots,n_{\\mathrm{fit}}$ (step 6)", "#f5e0dd")
box(6.3, 2.5, "plain eigensolve\nof fitted $V$ (step 7)", "#f5e0dd")
box(5.0, 0.8, "oracle: solve $N_{\\mathrm{sm}}+N_{\\mathrm{fl}} = n-\\frac{1}{2}$\n(Eq. (4), step 8)", "#efe6f5")
box(8.9, 2.5, "score held-out\nzeros only:\nrms, corr\nvs null", "#f5f0d5")

arrow(1.1, 3.75, 1.1, 3.0)
arrow(1.75, 4.2, 2.65, 4.2)
arrow(1.8, 2.5, 2.55, 2.5)
arrow(3.7, 3.75, 3.7, 3.0)
arrow(4.85, 2.5, 5.0, 3.85)
arrow(6.3, 3.75, 6.3, 2.95)
arrow(7.25, 2.5, 8.15, 2.5)
arrow(1.1, 2.0, 3.2, 1.0)
arrow(6.8, 0.8, 8.5, 1.9)
plt.savefig("paper-setup_pipeline.pdf", bbox_inches="tight")
