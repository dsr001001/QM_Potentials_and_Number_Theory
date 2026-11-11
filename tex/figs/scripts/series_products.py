import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
k = np.arange(2, 51)
p_plus = np.cumprod(1 + 1.0 / np.arange(1, 51))
p_sq = np.cumprod(1 - 1.0 / k**2)
p_minus = np.cumprod(1 - 1.0 / k)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
ax = axs[0]
ax.semilogy(np.arange(1, 51), p_plus, "o", ms=2, label=r"$\prod(1+1/k)$")
ax.semilogy(k, p_sq, "s", ms=2, label=r"$\prod(1-1/k^2)$")
ax.semilogy(k, p_minus, "^", ms=2, label=r"$\prod(1-1/k)$")
ax.axhline(0.5, color="gray", lw=0.8)
ax.set_xlabel(r"$n$ (dimensionless)")
ax.set_ylabel(r"partial product $P_n$")
ax.legend(fontsize=7)
ax = axs[1]
primes = np.array(qmnt.first_primes(100), dtype=float)
E = np.cumprod(1.0 / (1 - primes**-2))
ax.plot(np.arange(1, 101), E, "o-", ms=2, lw=0.8)
ax.axhline(np.pi**2 / 6, color="k", ls="--", lw=0.8)
ax.set_xlabel("number of primes used")
ax.set_ylabel(r"$\prod_{p\leq P}(1-p^{-2})^{-1}$")
plt.tight_layout()
plt.savefig("series_products.pdf", bbox_inches="tight")
