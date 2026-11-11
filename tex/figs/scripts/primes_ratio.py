import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expi
import qmnt

pr = np.asarray(qmnt.primes_upto(10**6))
x = np.logspace(1, 6, 500)
pi_x = np.searchsorted(pr, x, side="right")
Li = expi(np.log(x)) - expi(np.log(2.0))
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(x, pi_x / (x / np.log(x)), "b-", label=r"$\pi(x)\,/\,(x/\log x)$")
ax.semilogx(x, pi_x / Li, "r-", label=r"$\pi(x)\,/\,\mathrm{Li}(x)$")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"$x$ (dimensionless, log scale)")
ax.set_ylabel("ratio (dimensionless)")
ax.set_ylim(0.6, 1.3)
ax.legend(loc="lower right")
ax.grid(alpha=0.3, which="both")
plt.savefig("primes_ratio.pdf", bbox_inches="tight")
