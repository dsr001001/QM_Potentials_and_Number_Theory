import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

u = np.linspace(-4, 4, 801)
phi = np.exp(-u**2 / 2) / np.sqrt(2 * np.pi)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(u, phi, "k-", lw=2)
m2 = np.abs(u) <= 2
m1 = np.abs(u) <= 1
ax.fill_between(u[m2], 0, phi[m2], color="0.85", label=r"$|u|<2$: 95.4%")
ax.fill_between(u[m1], 0, phi[m1], color="0.55", label=r"$|u|<1$: 68.3%")
ax.set_xlabel(r"$u=(x-\mu)/\sigma$ (standard deviations)")
ax.set_ylabel(r"density $\varphi(u)$ (per standard deviation)")
ax.set_ylim(0, 0.45)
ax.legend(loc="upper right")
plt.savefig("probability_gaussian.pdf", bbox_inches="tight")
