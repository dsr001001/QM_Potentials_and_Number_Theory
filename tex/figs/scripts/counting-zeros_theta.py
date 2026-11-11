import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath as mp

t = np.linspace(3, 200, 300)
ex = np.array([float(mp.siegeltheta(x)) for x in t])
lead = t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8
asym = lead + 1 / (48 * t)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(t, ex, "C0", lw=1.6, label="exact")
a1.plot(t, asym, "C1--", lw=1.4, label="asymptotic")
a1.set_xlabel(r"$t$"); a1.set_ylabel(r"$\theta(t)$")
a1.legend(fontsize=8); a1.grid(alpha=0.3)
m = t >= 10
a2.plot(t[m], t[m] * (ex[m] - lead[m]), "C2", lw=1.5)
a2.axhline(1 / 48, color="gray", ls=":", label="1/48")
a2.set_xlabel(r"$t$"); a2.set_ylabel(r"$t\,[\theta-\mathrm{3\ terms}]$")
a2.legend(fontsize=8); a2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("counting-zeros_theta.pdf", bbox_inches="tight")
