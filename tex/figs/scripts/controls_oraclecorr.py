import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
import qmnt
n = np.arange(401, 501)
t = qmnt.zeros(500)[n - 1]
tp = qmnt.oracle_zeros(n, qmnt.first_primes(6))
ft, fp = qmnt.fluct(t, n), qmnt.fluct(tp, n)
B = 20000
idx = rng.integers(0, 100, size=(B, 100))
a = ft[idx] - ft[idx].mean(axis=1, keepdims=True)
b = fp[idx] - fp[idx].mean(axis=1, keepdims=True)
r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
lo, hi = np.percentile(r, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(r, bins=80, color="steelblue", edgecolor="none", density=True)
ax.axvline(lo, color="k", ls="--", lw=1)
ax.axvline(hi, color="k", ls="--", lw=1)
ax.set_xlabel("bootstrap held-out correlation (dimensionless)")
ax.set_ylabel("probability density")
ax.set_title(f"95% percentile interval [{lo:.3f}, {hi:.3f}]", fontsize=9)
plt.savefig("controls_oraclecorr.pdf", bbox_inches="tight")
