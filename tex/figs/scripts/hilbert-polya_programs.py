import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, s, ls="-"):
    ax.text(x, y, s, ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="k", ls=ls))
def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1))
ax.text(0.1, 5.4, "Analytic program", fontsize=9, weight="bold")
box(1.3, 4.5, "principles\n(primes, geometry,\nsymmetry)")
box(4.0, 4.5, "operator $H$\nspecified")
box(6.6, 4.5, "spectrum\ncomputed")
box(8.9, 4.5, "compare with\nall zeros:\nprediction")
for a, b in [(2.3, 3.2), (4.8, 5.8), (7.4, 8.0)]: arrow(a, b, 4.5)
ax.text(0.1, 2.5, "Empirical program", fontsize=9, weight="bold")
box(1.3, 1.5, "zeros\n$1,\\ldots,n_{\\rm fit}$")
box(4.0, 1.5, "fit parameters\nof a family")
box(6.6, 1.5, "in-sample\nagreement")
box(8.9, 1.5, "held-out\nzeros?", ls="--")
for a, b in [(2.2, 3.1), (4.9, 5.8), (7.4, 8.1)]: arrow(a, b, 1.5)
plt.savefig("hilbert-polya_programs.pdf", bbox_inches="tight")
