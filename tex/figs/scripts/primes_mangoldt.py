import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

K = 60
Lam = np.zeros(K + 1)
isprime = np.zeros(K + 1, dtype=bool)
for p in qmnt.primes_upto(K):
    p = int(p); isprime[p] = True
    q = p
    while q <= K:
        Lam[q] = np.log(p); q *= p
k = np.arange(1, K + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.vlines(k, 0, Lam[1:], color="gray", lw=0.8)
pm = isprime[1:]
hp = (Lam[1:] > 0) & (~pm)
ax.plot(k[pm], Lam[1:][pm], "o", color="C3", ms=5, label="primes: $\\Lambda(p)=\\log p$")
ax.plot(k[hp], Lam[1:][hp], "o", mfc="white", mec="C0", ms=6, label="higher prime powers $p^m$, $m\\geq 2$")
kk = np.linspace(1, K, 400)
ax.plot(kk, np.log(kk), "k--", lw=0.8, label="$\\log k$")
ax.set_xlabel("$k$ (integer)")
ax.set_ylabel("$\\Lambda(k)$ (dimensionless)")
ax.set_xlim(0, K + 1); ax.set_ylim(0, 4.3)
ax.legend(loc="upper left", fontsize=8)
plt.savefig("primes_mangoldt.pdf", bbox_inches="tight")
