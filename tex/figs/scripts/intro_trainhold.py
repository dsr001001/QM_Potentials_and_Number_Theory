import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

n = np.arange(370, 461)
tz = np.asarray(qmnt.zeros(460))
true = qmnt.fluct(tz[n - 1], n)
pred = qmnt.oracle_zeros(n, qmnt.first_primes(6))
predfl = qmnt.fluct(pred, n)

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(400.5, 460.5, color="gray", alpha=0.2, label="held out (not used in fitting)")
ax.plot(n, true, "o", color="k", ms=3.5, label="true fluctuation")
ax.plot(n, predfl, "-", color="tab:green", lw=1.4, label="six-prime explicit formula")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("zero index $n$")
ax.set_ylabel("fluctuation (mean spacings)")
ax.set_xlim(369, 461)
ax.legend(loc="lower left", fontsize=8)
plt.savefig("intro_trainhold.pdf", bbox_inches="tight")
