import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
def Nsm(t):
    u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def szero(n):
    return brentq(lambda t: Nsm(t) - (n - 0.5), 7.0, 1e6)
n = np.arange(1, 501)
s = np.array([szero(k) for k in n])
re2 = np.zeros_like(s)
for m in (1, 2, 3):
    re2 += np.sin(m*s*np.log(2))/(m*2**(m/2))
re2 = -re2/np.pi
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
sel = n >= 300
ax[0].plot(n[sel], -re2[sel], ".-", lw=0.8, ms=3)
ax[0].axvspan(400.5, 500.5, color="0.85")
ax[0].set_xlabel("level index n")
ax[0].set_ylabel("target fluctuation (mean spacings)")
ax[1].plot(n, np.log(s/(2*np.pi))/np.log(2), "k")
ax[1].set_xlabel("level index n")
ax[1].set_ylabel("levels per period of sin(t log 2)")
plt.tight_layout()
plt.savefig("paper-memorization_probe.pdf", bbox_inches="tight")
