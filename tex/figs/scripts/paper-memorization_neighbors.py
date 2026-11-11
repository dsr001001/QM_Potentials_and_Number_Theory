import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
import qmnt
def Nsm(t):
    u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def szero(n):
    return brentq(lambda t: Nsm(t) - (n - 0.5), 7.0, 1e6)
nmax = 62
t = np.asarray(qmnt.zeros(nmax), float)
n = np.arange(1, nmax + 1)
s = np.array([szero(k) for k in n])
g = 2*np.pi/np.log(s/(2*np.pi))
d = (t - s)/g
k = np.arange(2, 61)            # levels 2..60 have both neighbors
q = 0.5*(d[k - 2] + d[k])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n[:60], d[:60], "o-", color="C0", ms=4, lw=0.8, label="fluctuation d_n")
ax.plot(k, q, "s", mfc="none", color="C3", ms=5, label="neighbor average")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("level index n")
ax.set_ylabel("fluctuation (mean spacings)")
ax.legend(fontsize=8)
plt.savefig("paper-memorization_neighbors.pdf", bbox_inches="tight")
