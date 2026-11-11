import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
ds = np.diff(qmnt.Nsm(qmnt.zeros(10000)))
s = np.linspace(0, 3.5, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ds, bins=np.linspace(0, 3.5, 36), density=True, alpha=0.5, label="zeros 1-10^4")
ax.plot(s, 32/np.pi**2*s**2*np.exp(-4*s**2/np.pi), label="GUE")
ax.plot(s, np.pi/2*s*np.exp(-np.pi*s**2/4), ":", label="GOE")
ax.plot(s, np.exp(-s), "--", label="Poisson")
ax.set_xlabel("unfolded spacing s (mean spacings)"); ax.set_ylabel("p(s)"); ax.legend()
plt.savefig("level-statistics_gue.pdf", bbox_inches="tight")
