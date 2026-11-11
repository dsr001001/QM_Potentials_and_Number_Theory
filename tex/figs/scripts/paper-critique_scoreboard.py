import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Held-out rms on zeros 401--500 as reported in paper Tables I, IV, V
rows = [
    ("operator, cosine, M=200", 0.274, "tab:red"),
    ("operator, cosine, M=400", 0.369, "tab:red"),
    ("operator, M=400, ridge 0.03", 0.271, "tab:red"),
    ("operator, M=400, ridge 0.3", 0.265, "tab:red"),
    ("operator, cosine, M=600", 0.313, "tab:red"),
    ("C1: uniform grid, 900 coeff.", 0.258, "0.6"),
    ("B: 150 primes fitted (root-ambig.)", 0.180, "0.6"),
    ("oracle, 1 prime", 0.198, "tab:blue"),
    ("oracle, 3 primes", 0.116, "tab:blue"),
    ("oracle, 6 primes", 0.074, "tab:blue"),
    ("oracle, 25 primes", 0.043, "tab:blue"),
    ("A: oracle, 300 primes", 0.018, "tab:blue"),
    ("B: 25 primes, fitted", 0.028, "tab:blue"),
    ("C2: sparse search", 0.034, "tab:blue"),
]
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.invert_yaxis()
ax.axvline(0.263, color="k", ls="--", lw=1, label="null 0.263")
ax.set_xlabel("held-out rms on zeros 401-500 (mean spacings)")
ax.legend(loc="lower right", fontsize=8)
plt.savefig("paper-critique_scoreboard.pdf", bbox_inches="tight")
