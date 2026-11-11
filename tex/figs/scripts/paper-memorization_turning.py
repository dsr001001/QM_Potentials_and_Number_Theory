import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
V0 = 3.10073*np.pi
def xV(V):
    V = np.asarray(V, float); a = np.sqrt(np.maximum(V - V0, 0.0)); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b + a)/(b - a)))/np.pi
u = np.linspace(0, np.sqrt(1000 - V0), 4000)
Vg = V0 + u**2
xg = xV(Vg)
t = np.asarray(qmnt.zeros(500), float)
xt = xV(t)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].plot(xg, Vg, "k")
ax[0].axvspan(22.5, 28.5, color="C1", alpha=0.3, label="turning band, levels 201-300")
ax[0].plot([xV(500.0)], [500.0], "o", color="C3")
ax[0].set_xlabel("x (units hbar = 2m = 1)")
ax[0].set_ylabel("V(x) (energy units)")
ax[0].legend(fontsize=7, loc="upper left")
n = np.arange(1, 501)
ax[1].plot(n, xt, "k", lw=1)
ax[1].axvspan(201, 300, color="C1", alpha=0.3)
ax[1].set_xlabel("level index n")
ax[1].set_ylabel("turning point x(t_n)")
plt.tight_layout()
plt.savefig("paper-memorization_turning.pdf", bbox_inches="tight")
