import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = np.arange(1, 31)
S = np.cumsum((-1.0)**(n + 1) / n)
# rearranged series: 1/(2j-1) - 1/(4j-2) - 1/(4j)
terms = []
for j in range(1, 16):
    terms += [1.0 / (2 * j - 1), -1.0 / (4 * j - 2), -1.0 / (4 * j)]
R = np.cumsum(terms)
m = np.arange(1, len(R) + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(n, np.log(2) - 1.0 / (n + 1), np.log(2) + 1.0 / (n + 1),
                color="C0", alpha=0.15, label=r"$\log2\pm 1/(n+1)$")
ax.plot(n, S, "o-", ms=3, color="C0", label="alternating harmonic")
ax.axhline(np.log(2), color="C0", lw=0.8)
ax.plot(m, R, "s-", ms=3, color="C3", label="rearranged (one +, two -)")
ax.axhline(0.5 * np.log(2), color="C3", lw=0.8)
ax.text(31, np.log(2) + 0.01, r"$\log 2$", fontsize=9)
ax.text(31, 0.5 * np.log(2) + 0.01, r"$\frac{1}{2}\log 2$", fontsize=9)
ax.set_xlabel(r"number of terms $n$ (dimensionless)")
ax.set_ylabel(r"partial sum (dimensionless)")
ax.set_xlim(0, 46); ax.set_ylim(0.1, 1.05)
ax.legend(fontsize=8)
plt.savefig("series_alternating.pdf", bbox_inches="tight")
