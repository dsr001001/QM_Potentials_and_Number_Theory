import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(1000)); T = t[-1]
omega = np.linspace(0.3, 3.5, 6401)
spec = -np.array([np.cos(w*t).sum() for w in omega])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(omega, spec, lw=0.7, label=r"$-\sum_{n\leq 1000}\cos(\omega t_n)$")
for p in qmnt.primes_upto(33):
    q = p
    while q <= 33:
        w = np.log(q)
        if w >= 0.3:
            ax.plot(w, T/(2*np.pi)*np.log(p)/np.sqrt(q), 'ro', ms=4)
        q *= p
ax.plot([], [], 'ro', ms=4, label="Landau prediction")
ax.set_xlabel(r"angular frequency $\omega$ (radians per unit t)")
ax.set_ylabel("spectrum (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("explicit-formula_landau.pdf", bbox_inches="tight")
