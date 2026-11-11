import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
p = np.asarray(qmnt.primes_upto(10**6), float)
fig, ax = plt.subplots(figsize=(6, 4))
for s, lab in [(2, r"$\sigma=2$"), (1, r"$\sigma=1$"), (0.5, r"$\sigma=1/2$")]:
    ax.loglog(p, np.cumsum(p**-s), label=lab)
x = np.logspace(1, 6, 200)
ax.loglog(x, 2*np.sqrt(x)/np.log(x), 'k--', lw=1, label=r"$2\sqrt{x}/\log x$")
ax.set_xlabel("x (dimensionless)")
ax.set_ylabel(r"$\sum_{p\leq x} p^{-\sigma}$ (dimensionless)")
ax.legend()
plt.savefig("explicit-formula_partial.pdf", bbox_inches="tight")
