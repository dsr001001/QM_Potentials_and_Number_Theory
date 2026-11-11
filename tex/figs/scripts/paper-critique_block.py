import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(401, 501)
t = qmnt.zeros(500)[n - 1]
f = qmnt.fluct(t, n)
g = qmnt.fluct(qmnt.oracle_zeros(n, qmnt.first_primes(6)), n)
rng = np.random.default_rng(1)
blocks = [1, 2, 3, 5, 8, 10, 15, 20]
lo, hi = [], []
N = len(f)
for Lb in blocks:
    nb = int(np.ceil(N / Lb))
    vals = np.empty(4000)
    for b in range(4000):
        starts = rng.integers(0, N, size=nb)
        idx = ((starts[:, None] + np.arange(Lb)[None, :]).ravel()[:N]) % N
        vals[b] = np.corrcoef(f[idx], g[idx])[0, 1]
    l, h = np.percentile(vals, [2.5, 97.5])
    lo.append(l); hi.append(h)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axhspan(0.95, 0.97, color="0.85", label="paper [+0.95, +0.97]")
ax.plot(blocks, lo, "bo-", label="2.5th percentile")
ax.plot(blocks, hi, "ro-", label="97.5th percentile")
ax.set_xlabel("block length (levels)")
ax.set_ylabel("held-out correlation, 6-prime oracle")
ax.legend(fontsize=8)
plt.savefig("paper-critique_block.pdf", bbox_inches="tight")
