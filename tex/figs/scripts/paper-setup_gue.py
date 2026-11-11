import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def Nsm(t):
    u = t / (2*np.pi)
    return u*np.log(u) - u + 7/8

t = np.asarray(qmnt.zeros(10000), float)
sp = np.diff(Nsm(t))
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(sp, bins=np.linspace(0, 3.5, 50), density=True, color="0.75",
        edgecolor="0.4", label="first $10^4$ zeros")
s = np.linspace(0, 3.5, 400)
ax.plot(s, 32/np.pi**2 * s**2 * np.exp(-4*s**2/np.pi), "b-",
        label="GUE surmise")
ax.plot(s, np.exp(-s), "r--", label="Poisson $e^{-s}$")
ax.set_xlabel("unfolded spacing $s$ (units of mean spacing)")
ax.set_ylabel("probability density")
ax.legend()
plt.savefig("paper-setup_gue.pdf", bbox_inches="tight")
