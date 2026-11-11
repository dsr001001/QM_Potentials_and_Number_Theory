import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expi
import qmnt

pr = np.asarray(qmnt.primes_upto(100))
x = np.linspace(2, 100, 3000)
pi_x = np.searchsorted(pr, x, side="right")
Li = expi(np.log(x)) - expi(np.log(2.0))      # integral from 2 to x of dt/log t
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(x, pi_x, where="post", color="k", lw=1.5, label=r"$\pi(x)$")
ax.plot(x, x / np.log(x), "b--", label=r"$x/\log x$")
ax.plot(x, Li, "r-", lw=1, label=r"$\mathrm{Li}(x)=\int_2^x dt/\log t$")
ax.set_xlabel(r"$x$ (dimensionless)")
ax.set_ylabel("number of primes up to $x$")
ax.set_xlim(0, 100); ax.set_ylim(0, 31)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.savefig("primes_pi_small.pdf", bbox_inches="tight")
