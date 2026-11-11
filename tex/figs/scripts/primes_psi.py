import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

X = 10**5
Lam = np.zeros(X + 1); Th = np.zeros(X + 1)
for p in qmnt.primes_upto(X):
    p = int(p); Th[p] = np.log(p)
    q = p
    while q <= X:
        Lam[q] = np.log(p); q *= p
psi = np.cumsum(Lam); theta = np.cumsum(Th)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
n = np.arange(0, 101)
a1.step(n, psi[:101], where="post", color="k", label=r"$\psi(x)$")
a1.step(n, theta[:101], where="post", color="C0", lw=0.9, label=r"$\theta(x)$")
a1.plot([0, 100], [0, 100], "r--", lw=0.8, label="$y=x$")
a1.set_xlabel("$x$ (dimensionless)"); a1.set_ylabel("value (dimensionless)")
a1.set_title("(a)", fontsize=9); a1.legend(fontsize=7, loc="upper left")
xs = np.arange(2, X + 1)
a2.plot(xs, (psi[2:] - xs) / np.sqrt(xs), color="k", lw=0.4)
a2.axhline(0, color="r", lw=0.6)
a2.set_xscale("log")
a2.set_xlabel("$x$ (log scale)"); a2.set_ylabel(r"$(\psi(x)-x)/\sqrt{x}$")
a2.set_title("(b)", fontsize=9)
plt.tight_layout()
plt.savefig("primes_psi.pdf", bbox_inches="tight")
