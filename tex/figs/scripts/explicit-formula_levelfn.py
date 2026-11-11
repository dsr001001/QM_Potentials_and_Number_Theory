import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
from scipy.optimize import brentq

def Nsm(t):
    return t / (2 * np.pi) * np.log(t / (2 * np.pi)) - t / (2 * np.pi) + 7 / 8

P = qmnt.first_primes(6)
def Nfl(t):
    t = np.asarray(t, float); s = np.zeros_like(t)
    for p in P:
        for m in range(1, 4):
            s = s + np.sin(m * t * np.log(p)) / (m * p ** (m / 2))
    return -s / np.pi

z = qmnt.zeros(12)
t = np.linspace(10, 52, 6000)
F = Nsm(t) + Nfl(t)
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(t, np.searchsorted(z, t, side="right"), "k-", where="post", lw=1, label="N(t)")
ax.plot(t, Nsm(t), "-", color="gray", lw=1, label=r"$N_{\rm sm}$")
ax.plot(t, F, "-", lw=2, label=r"$N_{\rm sm}+N_{\rm fl}$ (6 primes)")
for n in range(1, 11):
    ax.axhline(n - 0.5, ls="--", color="gray", lw=0.4)
    tt = t[t < 52]; ff = F[t < 52] - (n - 0.5)
    idx = np.where(np.sign(ff[:-1]) != np.sign(ff[1:]))[0]
    for i in idx:
        r = brentq(lambda x: float(Nsm(x) + Nfl(x)) - (n - 0.5), tt[i], tt[i + 1])
        ax.plot(r, n - 0.5, "x", color="C3", ms=6)
    if z[n - 1] < 52:
        ax.plot(z[n - 1], n - 0.5, "o", mfc="none", color="k", ms=6)
ax.set_xlabel("height t (dimensionless)")
ax.set_ylabel("count (dimensionless)")
ax.set_xlim(10, 52)
ax.legend(fontsize=8, loc="upper left")
plt.savefig("explicit-formula_levelfn.pdf", bbox_inches="tight")
