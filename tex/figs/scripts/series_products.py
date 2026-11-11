import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = np.arange(2, 51).astype(float)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.semilogy(n, n + 1, 'o', ms=3, label=r'$\prod(1+1/k)$')
a1.semilogy(n, (n + 1)/(2*n), 's', ms=3, label=r'$\prod(1-1/k^2)$')
a1.semilogy(n, 1/n, '^', ms=3, label=r'$\prod(1-1/k)$')
a1.set_xlabel('number of factors $n$ (dimensionless)')
a1.set_ylabel('partial product (dimensionless)')
a1.legend(fontsize=7)
sieve = np.ones(600, bool); sieve[:2] = False
for i in range(2, 25):
    if sieve[i]:
        sieve[i*i::i] = False
primes = np.nonzero(sieve)[0][:100].astype(float)
P = np.cumprod(primes**2/(primes**2 - 1))
a2.plot(np.arange(1, 101), P, '.-', ms=3, lw=0.6)
a2.axhline(np.pi**2/6, color='k', ls='--', lw=0.8, label=r'$\pi^2/6$')
a2.set_xlabel('number of primes used (dimensionless)')
a2.set_ylabel('partial Euler product at $s=2$')
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("series_products.pdf", bbox_inches="tight")
