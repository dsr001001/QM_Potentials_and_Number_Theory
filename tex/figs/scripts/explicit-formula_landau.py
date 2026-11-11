import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = qmnt.zeros(1000)
T = t[-1]
omega = np.linspace(0.3, 3.5, 6401)
spec = np.empty_like(omega)
for i in range(0, len(omega), 400):
    w = omega[i:i + 400]
    spec[i:i + 400] = -np.cos(np.outer(w, t)).sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(omega, spec, "-", lw=0.8, label=r"$-\sum_{n\leq 1000}\cos(\omega t_n)$")
first = True
for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
    m = 1
    while p ** m <= np.exp(3.5):
        k = p ** m
        if np.log(k) >= 0.3:
            ax.plot(np.log(k), T / (2 * np.pi) * np.log(p) / np.sqrt(k), "o",
                    color="C3", ms=4, label="Landau prediction" if first else None)
            first = False
        m += 1
ax.set_xlabel(r"angular frequency $\omega$ (rad per unit t)")
ax.set_ylabel("spectral amplitude (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("explicit-formula_landau.pdf", bbox_inches="tight")
