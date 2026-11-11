import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 3))
n = np.arange(1, 501)
rows = []
rows.append(("forward", n <= 400, (n >= 401)))
rng = np.random.default_rng(1)
test = rng.choice(400, 80, replace=False) + 1
tr = np.zeros(500, bool); te = np.zeros(500, bool)
tr[:400] = True; tr[test - 1] = False; te[test - 1] = True
rows.append(("random 80/20", tr, te))
rows.append(("odd/even", (n <= 400) & (n % 2 == 1), (n <= 400) & (n % 2 == 0)))
rows.append(("interpolation", (n <= 200) | (n >= 301), (n >= 201) & (n <= 300)))
for i, (name, trm, tem) in enumerate(rows):
    y = 3 - i
    ax.vlines(n[trm], y - 0.3, y + 0.3, color="tab:blue", lw=0.6)
    ax.vlines(n[tem], y - 0.3, y + 0.3, color="tab:orange", lw=0.6)
    ax.text(-10, y, name, ha="right", va="center", fontsize=8)
ax.set_xlim(-130, 510); ax.set_ylim(-0.6, 3.6)
ax.set_yticks([]); ax.set_xlabel("zero index $n$ (scale 1)")
ax.plot([], [], color="tab:blue", label="training"); ax.plot([], [], color="tab:orange", label="held out")
ax.legend(fontsize=8, loc="upper right", ncol=2)
plt.savefig("paper-setup_splits.pdf", bbox_inches="tight")
