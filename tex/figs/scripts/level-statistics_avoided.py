import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tau = np.linspace(-0.3, 0.3, 400)
g = 0.05
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tau, tau, "k--", lw=1, label="g = 0 (crossing)")
ax.plot(tau, -tau, "k--", lw=1)
ax.plot(tau, np.sqrt(tau**2 + g**2), "b-", lw=2, label="g = 0.05 (avoided)")
ax.plot(tau, -np.sqrt(tau**2 + g**2), "b-", lw=2)
ax.annotate("", xy=(0, g), xytext=(0, -g), arrowprops=dict(arrowstyle="<->"))
ax.text(0.01, 0.0, "2g = 0.1", va="center")
ax.set_xlabel(r"control parameter $\tau$ (dimensionless)")
ax.set_ylabel("eigenvalue (dimensionless)")
ax.legend()
plt.savefig("level-statistics_avoided.pdf", bbox_inches="tight")
