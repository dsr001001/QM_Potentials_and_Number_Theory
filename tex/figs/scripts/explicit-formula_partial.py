import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

p = np.asarray(qmnt.primes_upto(100000), dtype=float)
fig, ax = plt.subplots(figsize=(6, 4))
for sig, st, lab in [(2.0, "-", r"$\sigma=2$"), (1.0, "--", r"$\sigma=1$"),
                     (0.5, ":", r"$\sigma=1/2$")]:
    ax.plot(p, np.cumsum(p ** (-sig)), st, lw=1.8, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("prime cutoff x (dimensionless)")
ax.set_ylabel(r"$\sum_{p\leq x} p^{-\sigma}$")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("explicit-formula_partial.pdf", bbox_inches="tight")
