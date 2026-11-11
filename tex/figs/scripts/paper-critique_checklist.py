import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
items = [
    ("(i) train/test split vs own null,\n     incl. a within-height split", "Tables I and III, Fig. 1"),
    ("(ii) score the exact operator\n     spectrum, not a surrogate", "Sec. V, Fig. 5"),
    ("(iii) survives regularization\n     and optimizer restarts", "Table I ridge rows and repeats"),
    ("(iv) refit to permuted and\n     phase-randomized targets", "Sec. III surrogate tests"),
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
for i, (item, why) in enumerate(items):
    yc = 8.5 - 2.2 * i
    ax.add_patch(plt.Rectangle((0.2, yc - 0.35), 0.7, 0.7, fill=False, lw=1.5))
    ax.plot([0.35, 0.55, 0.85], [yc, yc - 0.25, yc + 0.3], color="tab:green", lw=2)
    ax.text(1.2, yc, item, va="center", fontsize=9)
    ax.annotate("", xy=(6.4, yc), xytext=(5.6, yc), arrowprops=dict(arrowstyle="<-", color="0.4"))
    ax.text(6.6, yc, why, va="center", fontsize=8, color="0.25")
ax.text(1.2, 9.7, "item of the standard", fontsize=9, weight="bold")
ax.text(6.6, 9.7, "experiment motivating it", fontsize=9, weight="bold")
plt.savefig("paper-critique_checklist.pdf", bbox_inches="tight")
