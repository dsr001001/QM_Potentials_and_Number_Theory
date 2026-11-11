import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
P = list(qmnt.first_primes(6))
z = np.asarray(qmnt.zeros(14))
t = np.linspace(10, 60, 5000)
Nsm = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
fl = np.zeros_like(t)
for p in P:
    for m in (1, 2, 3):
        fl += np.sin(m*t*np.log(p))/(m*p**(m/2))
fl = -fl/np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(t, np.searchsorted(z, t, side="right"), 'k', lw=0.8, where="post", label="N(t)")
ax.plot(t, Nsm, lw=0.8, label=r"$N_{\rm sm}$")
ax.plot(t, Nsm + fl, lw=2, label=r"$N_{\rm sm}+N_{\rm fl}$ (6 primes)")
n = np.arange(1, 13)
for v in n - 0.5:
    ax.axhline(v, ls="--", lw=0.4, color="gray")
zt = z[:12]; ax.plot(zt, n - 0.5, 'o', mfc="none", color="k", label="true zeros")
tp = np.asarray(qmnt.oracle_zeros(n, P, mmax=3)); ax.plot(tp, n - 0.5, 'x', color="r", label="predicted")
ax.set_xlabel("height t (dimensionless)"); ax.set_ylabel("count (dimensionless)")
ax.legend(fontsize=8, loc="upper left")
plt.savefig("explicit-formula_levelfn.pdf", bbox_inches="tight")
