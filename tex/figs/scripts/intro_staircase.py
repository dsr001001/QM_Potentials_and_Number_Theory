import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

tz = np.asarray(qmnt.zeros(40))
T = np.linspace(8.0, 100.0, 4000)
N = np.searchsorted(tz, T, side="right")
u = T / (2 * np.pi)
Nsm = u * np.log(u) - u + 7.0 / 8.0

fig, ax = plt.subplots(figsize=(6, 4))
ax.step(T, N, where="post", color="k", lw=1.2, label=r"staircase $N(T)$")
ax.plot(T, Nsm, color="tab:blue", lw=1.5, label=r"smooth $N_{\rm sm}(T)$, Eq. (1)")
ax.set_xlabel(r"height $T$ (dimensionless)")
ax.set_ylabel("number of zeros with $t \\leq T$")
ax.set_xlim(8, 100)
ax.set_ylim(-0.5, 30)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.savefig("intro_staircase.pdf", bbox_inches="tight")
