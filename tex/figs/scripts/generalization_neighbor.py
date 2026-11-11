import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(400); n = np.arange(1, 401); d = qmnt.fluct(t, n)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
k = np.arange(1, 61)
a1.plot(k, d[:60], "k.-", lw=0.8, label=r"$d_n$")
ev = np.arange(2, 60, 2)
pred = 0.5*(d[ev-2] + d[ev])
a1.plot(ev, pred, "rs", ms=3, label="neighbor mean")
a1.axhline(0, color="0.6", lw=0.5)
a1.set_xlabel(r"zero index $n$"); a1.set_ylabel("fluctuation (mean spacings)")
a1.legend(fontsize=7)
dc = d - d.mean(); lags = np.arange(1, 21)
acf = [np.sum(dc[:-L]*dc[L:])/np.sum(dc**2) for L in lags]
a2.bar(lags, acf, color="0.4")
a2.axhline(0, color="k", lw=0.5)
a2.set_xlabel(r"lag $k$ (levels)"); a2.set_ylabel(r"$\rho_k$ (dimensionless)")
plt.tight_layout()
plt.savefig("generalization_neighbor.pdf", bbox_inches="tight")
