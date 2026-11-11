import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(401, 501)
t = qmnt.zeros(500)[n - 1]
tp = qmnt.oracle_zeros(n, qmnt.first_primes(6))
f = np.asarray(qmnt.fluct(t, n))
g = np.asarray(qmnt.fluct(tp, n))
N = len(f)
rng = np.random.default_rng(1)
blocks = [1, 2, 3, 5, 10, 20]
B = 3000
lo, hi, mid = [], [], []
for b in blocks:
    nb = int(np.ceil(N / b))
    cs = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, N, size=nb)
        idx = ((starts[:, None] + np.arange(b)[None, :]).ravel()[:N]) % N
        cs[i] = np.corrcoef(f[idx], g[idx])[0, 1]
    l, h = np.percentile(cs, [2.5, 97.5])
    lo.append(l); hi.append(h); mid.append(np.median(cs))
lo, hi, mid = map(np.array, (lo, hi, mid))
fig, ax = plt.subplots(figsize=(6, 4))
ax.axhspan(0.95, 0.97, color="0.85", label="paper [+0.95, +0.97]")
ax.errorbar(blocks, mid, yerr=[mid - lo, hi - mid], fmt="o", capsize=4, color="tab:blue", label="qmnt, circular block bootstrap")
ax.set_xscale("log")
ax.set_xticks(blocks); ax.set_xticklabels([str(b) for b in blocks])
ax.set_xlabel("block length (levels)")
ax.set_ylabel("held-out correlation, 6-prime oracle")
ax.legend(fontsize=8, loc="lower left")
plt.savefig("paper-critique_block.pdf", bbox_inches="tight")
