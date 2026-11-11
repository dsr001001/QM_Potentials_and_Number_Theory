import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

n = np.arange(1, 1001)
t = np.asarray(qmnt.zeros(1000))
g = qmnt.mean_gap(n)
d = qmnt.fluct(t, n)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(n, g, "C0")
a1.set_xlabel("zero index $n$"); a1.set_ylabel(r"gap$_n$ (units of $t$)")
a1.grid(alpha=0.3)
a2.axvspan(401, 500, color="orange", alpha=0.3)
a2.plot(n, d, "C0.", ms=1.5)
a2.set_xlabel("zero index $n$"); a2.set_ylabel(r"$\delta_n$ (mean spacings)")
a2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("counting-zeros_fluct.pdf", bbox_inches="tight")
