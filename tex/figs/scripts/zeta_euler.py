import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]

p = primes_upto(100).astype(float)
E2 = np.cumprod(1.0 / (1.0 - p**-2))

xmax = 1000
x = np.arange(1, xmax + 1)
H = np.cumsum(1.0 / x)
fac = np.ones(xmax + 1)
pp = primes_upto(xmax)
fac[pp] = 1.0 / (1.0 - 1.0 / pp.astype(float))
E1 = np.cumprod(fac[1:])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.step(p, E2, where="post", color="k", label=r"$E_P(2)$")
a1.plot(p, E2, "o", ms=2.5, color="k")
a1.axhline(np.pi**2 / 6, color="C3", ls="--", label=r"$\pi^2/6$")
a1.set_xlabel(r"largest prime included $P$ (dimensionless)")
a1.set_ylabel("value (dimensionless)")
a1.set_title("(a) $s=2$", fontsize=9)
a1.legend(fontsize=7, loc="lower right")

a2.plot(x, E1, "k", label=r"$\prod_{p\leq x}(1-1/p)^{-1}$")
a2.plot(x, H, "--", color="C0", label=r"$\sum_{k\leq x}1/k$")
a2.set_xscale("log")
a2.set_xlabel(r"$x$ (dimensionless)")
a2.set_ylabel("value (dimensionless)")
a2.set_title("(b) $s=1$", fontsize=9)
a2.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig("zeta_euler.pdf", bbox_inches="tight")
