import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lam = np.linspace(-0.5, 0.5, 400); g = 0.05
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lam, lam, "k--", lam, -lam, "k--", lw=1)
ax.plot(lam, np.sqrt(lam**2 + g**2), "b", lam, -np.sqrt(lam**2 + g**2), "b")
ax.set_xlabel("parameter lambda (dimensionless)"); ax.set_ylabel("eigenvalue (dimensionless)")
plt.savefig("level-statistics_avoided.pdf", bbox_inches="tight")
