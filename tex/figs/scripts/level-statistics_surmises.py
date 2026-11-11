import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s = np.linspace(0, 3.5, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(s, np.exp(-s), "--", label="Poisson")
ax.plot(s, np.pi/2*s*np.exp(-np.pi*s**2/4), ":", label="GOE")
ax.plot(s, 32/np.pi**2*s**2*np.exp(-4*s**2/np.pi), label="GUE")
ax.set_xlabel("unfolded spacing s (mean spacings)"); ax.set_ylabel("p(s)"); ax.legend()
plt.savefig("level-statistics_surmises.pdf", bbox_inches="tight")
