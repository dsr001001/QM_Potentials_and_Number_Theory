import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Values quoted from paper Tables I, IV, V (fit 1-400, held-out 401-500)
rows = [
    ("operator, cosine, M=400", 0.369, "r"),
    ("operator, cosine, M=600", 0.313, "r"),
    ("C1: dense grid, 900 coeffs", 0.258, "0.5"),
    ("B: 150 primes fitted (root-ambig.)", 0.180, "0.5"),
    ("oracle, prime 2 only", 0.198, "b"),
    ("oracle, 6 primes", 0.074, "b"),
    ("oracle, 25 primes", 0.043, "b"),
    ("C2: sparse search (72 params)", 0.034, "b"),
    ("B: 25 primes, amplitudes fitted", 0.028, "b"),
    ("A: oracle, 300 primes", 0.018, "b"),
]
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, c) in zip(y, rows):
    ax.barh(yi, v, color=c, alpha=0.8)
    ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.axvline(0.263, ls="--", color="k")
ax.text(0.266, y[0] + 0.3, "null 0.263", fontsize=7)
ax.set_xlim(0, 0.42)
ax.set_xlabel("held-out rms on zeros 401-500 (mean spacings)")
ax.set_title("Paper Tables I, IV, V: fit on zeros 1-400", fontsize=9)
plt.savefig("paper-critique_scoreboard.pdf", bbox_inches="tight")
