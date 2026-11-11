import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def Nsm(t):
    return t / (2 * np.pi) * np.log(t / (2 * np.pi)) - t / (2 * np.pi) + 7 / 8

def Nfl(t, P, mmax=3):
    s = np.zeros_like(t)
    for p in P:
        for m in range(1, mmax + 1):
            s += np.sin(m * t * np.log(p)) / (m * p ** (m / 2))
    return -s / np.pi

z = qmnt.zeros(60)
t = np.linspace(10, 100, 20000)
N = np.searchsorted(z, t, side="right")
S = N - Nsm(t)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, S, "k-", lw=1, label=r"$N(t)-N_{\rm sm}(t)$")
ax.plot(t, Nfl(t, [2]), "--", lw=1.3, label="primes {2}")
ax.plot(t, Nfl(t, qmnt.first_primes(6)), "-", lw=1.5, label="6 primes")
ax.plot(t, Nfl(t, qmnt.first_primes(25)), "-", lw=1.0, alpha=0.8, label="25 primes")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("height t (dimensionless)")
ax.set_ylabel("fluctuating count (mean spacings)")
ax.legend(fontsize=8, loc="lower right")
plt.savefig("explicit-formula_staircase.pdf", bbox_inches="tight")
