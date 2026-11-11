import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
z = qmnt.zeros(30)
t = np.linspace(10.0, 60.0, 3000)
N = np.searchsorted(z, t, side="right")
Nsm = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
Nfl = np.zeros_like(t)
for p in [2, 3, 5, 7, 11, 13]:
    for m in range(1, 4):
        Nfl -= np.sin(m*t*np.log(p)) / (m * p**(m/2)) / np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, N, color="k", lw=1.3, label="$N(t)$ (true count)")
ax.plot(t, Nsm, "k--", lw=0.9, label=r"$N_{\rm sm}(t)$")
ax.plot(t, Nsm + Nfl, color="C3", lw=1.1, label=r"$N_{\rm sm}+N_{\rm fl}$, $p\leq 13$, $m\leq 3$")
ax.set_xlabel("height $t$ (dimensionless)")
ax.set_ylabel("number of zeros")
ax.legend(fontsize=8, loc="upper left")
plt.savefig("hilbert-polya_zetastair.pdf", bbox_inches="tight")
