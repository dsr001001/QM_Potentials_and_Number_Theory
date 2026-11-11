import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def Nsm(t):
    u = t / (2 * np.pi)
    return u * np.log(u) - u + 7 / 8

t = np.asarray(qmnt.zeros(60))
tz = t[t <= 100]
T = np.linspace(1.0, 100.0, 5000)
N = np.searchsorted(t, T, side="right")

fig, ax = plt.subplots(figsize=(6, 4))
ax.step(T, N, where="post", color="C0", lw=1.4, label=r"$N(T)$, true count")
ax.plot(T, Nsm(T), "--", color="C1", lw=1.6, label=r"$N_{\rm sm}(T)$, Eq. (1)")
ax.plot(tz, np.arange(1, len(tz) + 1), "k.", ms=4, label=r"zeros $(t_n, n)$")
ax.set_xlabel(r"height $T$ (dimensionless)")
ax.set_ylabel("number of zeros")
ax.set_xlim(0, 100)
ax.set_ylim(-1, 31)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
plt.savefig("counting-zeros_staircase.pdf", bbox_inches="tight")
