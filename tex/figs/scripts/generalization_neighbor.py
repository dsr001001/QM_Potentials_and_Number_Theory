import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = qmnt.zeros(400); n = np.arange(1, 401)
d = qmnt.fluct(t, n)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
m = 60
a1.plot(n[:m], d[:m], "k.-", lw=0.8, label=r"true $d_n$")
ev = np.arange(1, m - 1, 2)
pred = 0.5*(d[ev - 1] + d[ev + 1])
a1.plot(n[ev], pred, "s", color="C3", ms=4, label="neighbor mean (even $n$)")
a1.axhline(0, color="0.5", lw=0.5)
a1.set_xlabel("level index $n$")
a1.set_ylabel(r"fluctuation $d_n$ (mean spacings)")
a1.legend(fontsize=7)

dc = d - d.mean()
lags = np.arange(0, 13)
acf = [1.0] + [np.sum(dc[:-k]*dc[k:])/np.sum(dc**2) for k in lags[1:]]
a2.bar(lags, acf, color="C0")
a2.axhline(0, color="k", lw=0.5)
a2.set_xlabel("lag $k$ (levels)")
a2.set_ylabel(r"autocorrelation $\rho_k$ (dimensionless)")
plt.tight_layout()
plt.savefig("generalization_neighbor.pdf", bbox_inches="tight")
