import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
z = np.asarray(qmnt.zeros(60))
t = np.linspace(10, 100, 8000)
Nsm = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
S = np.searchsorted(z, t, side="right") - Nsm
def Nfl(t, P):
    s = np.zeros_like(t)
    for p in P:
        for m in (1, 2, 3):
            s += np.sin(m*t*np.log(p))/(m*p**(m/2))
    return -s/np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, S, 'k', lw=1, label=r"$N(t)-N_{\rm sm}(t)$")
for k, c in [(1, "tab:blue"), (6, "tab:orange"), (25, "tab:green")]:
    ax.plot(t, Nfl(t, list(qmnt.first_primes(k))), color=c, lw=1.3, label=f"{k} prime(s)")
ax.set_xlabel("height t (dimensionless)")
ax.set_ylabel("count (mean spacings)")
ax.legend(fontsize=8)
plt.savefig("explicit-formula_staircase.pdf", bbox_inches="tight")
