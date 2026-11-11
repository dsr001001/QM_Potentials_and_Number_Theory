import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

N = 100
spf = np.zeros(N + 1, dtype=int)        # smallest prime factor
for k in range(2, N + 1):
    if spf[k] == 0:
        for j in range(k, N + 1, k):
            if spf[j] == 0:
                spf[j] = k
colors = {2: "#d9d9d9", 3: "#c6dbef", 5: "#fdd0a2", 7: "#c7e9c0"}
fig, ax = plt.subplots(figsize=(6, 4))
for k in range(1, N + 1):
    r = (k - 1) // 10
    c = (k - 1) % 10
    if k == 1:
        fc = "white"
    elif spf[k] == k:
        fc = "#e34a33"
    else:
        fc = colors[spf[k]]
    ax.add_patch(Rectangle((c, 9 - r), 0.95, 0.95, facecolor=fc, edgecolor="k", lw=0.5))
    txtc = "white" if (k > 1 and spf[k] == k) else "black"
    ax.text(c + 0.475, 9 - r + 0.475, str(k), ha="center", va="center", fontsize=7, color=txtc)
ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.1, 10.1)
ax.set_aspect("equal"); ax.axis("off")
handles = [Patch(facecolor="#e34a33", edgecolor="k", label="prime"),
           Patch(facecolor=colors[2], edgecolor="k", label="crossed out by 2"),
           Patch(facecolor=colors[3], edgecolor="k", label="crossed out by 3"),
           Patch(facecolor=colors[5], edgecolor="k", label="crossed out by 5"),
           Patch(facecolor=colors[7], edgecolor="k", label="crossed out by 7"),
           Patch(facecolor="white", edgecolor="k", label="1 (neither)")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
ax.set_title("Sieve of Eratosthenes on 1-100", fontsize=10)
plt.savefig("primes_sieve.pdf", bbox_inches="tight")
