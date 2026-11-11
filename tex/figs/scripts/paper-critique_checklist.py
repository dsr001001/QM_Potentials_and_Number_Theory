import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10); ax.set_ylim(-0.3, 10); ax.axis("off")
items = [
    (9.3, "Claim: a fitted operator 'captures' the zeros", ""),
    (7.6, "(i) Train/test split: held-out rms, corr\nvs own smooth null; within-height split too",
     "Fig. 1, Tables I, III"),
    (5.8, "(ii) Score the exact operator spectrum,\nnot a surrogate", "Sec. V, Fig. 5"),
    (4.0, "(iii) Skill survives regularization\nand optimizer restarts", "Table I ridge rows,\nrepeats in caption"),
    (2.2, "(iv) Refit to permuted and phase-\nrandomized targets: zeros must do better", "Sec. III surrogates"),
    (0.4, "Passes the minimal standard\n(necessary, not sufficient)", ""),
]
for y, txt, note in items:
    ax.text(3.6, y, txt, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="k"))
    if note:
        ax.text(7.4, y, note, ha="left", va="center", fontsize=7.5, color="0.3")
for i in range(len(items) - 1):
    ax.annotate("", xy=(3.6, items[i + 1][0] + 0.62), xytext=(3.6, items[i][0] - 0.62),
                arrowprops=dict(arrowstyle="->"))
ax.text(7.4, 9.3, "motivated by:", fontsize=7.5, color="0.3", va="center")
plt.savefig("paper-critique_checklist.pdf", bbox_inches="tight")
