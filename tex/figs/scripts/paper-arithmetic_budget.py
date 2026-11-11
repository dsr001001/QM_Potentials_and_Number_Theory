import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = [("operator M=400 (400)", 0.369, None, "tab:red"),
        ("operator M=600 (600)", 0.313, None, "tab:red"),
        ("A 25 primes (0)", 0.043, None, "tab:green"),
        ("A 300 primes (0)", 0.018, (0.015, 0.020), "tab:green"),
        ("B 25 primes (150)", 0.028, None, "tab:green"),
        ("B 150 primes (900)", 0.180, (0.158, 0.201), "tab:green"),
        ("C1 grid (900)", 0.258, (0.231, 0.285), "tab:red"),
        ("C2 sparse (72)", 0.034, (0.028, 0.041), "tab:blue")]
fig, ax = plt.subplots(figsize=(6, 4))
for i, (nm, v, ci, c) in enumerate(rows):
    ax.plot(v, i, "o", color=c)
    if ci:
        ax.plot(ci, [i, i], color=c, lw=2)
ax.axvline(0.263, color="k", ls="--", label="null 0.263")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("held-out rms, zeros 401-500 (mean spacings)")
ax.legend(fontsize=8)
plt.savefig("paper-arithmetic_budget.pdf", bbox_inches="tight")
