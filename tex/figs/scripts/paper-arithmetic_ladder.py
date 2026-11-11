import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

names = ["A: 300 primes (0)", "B: 25 primes (150)", "C2: sparse (72)",
         "A: 25 primes (0)", "B: 150 primes (900)", "C1: grid (900)",
         "operator M=600 (600)", "operator M=400 (400)"]
vals = [0.018, 0.028, 0.034, 0.043, 0.180, 0.258, 0.313, 0.369]
cols = ["tab:green", "tab:green", "tab:blue", "tab:green", "tab:green",
        "tab:red", "tab:red", "tab:red"]
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(names))
ax.barh(y, vals, color=cols)
ax.axvline(0.263, color="k", ls="--", label="null 0.263")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("held-out rms, zeros 401-500 (mean spacings)")
ax.set_title("green: (i) arithmetic, blue: (ii) sparsity, red: (iii) generic", fontsize=8)
ax.legend(fontsize=8)
plt.savefig("paper-arithmetic_ladder.pdf", bbox_inches="tight")
