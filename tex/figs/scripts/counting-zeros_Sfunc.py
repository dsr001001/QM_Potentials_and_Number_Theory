import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = np.asarray(qmnt.zeros(200))
T = np.linspace(10, 300, 20000)
N = np.searchsorted(t, T, side="right")
u = T / (2 * np.pi)
fl = N - (u * np.log(u) - u + 7 / 8)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(T, fl, "C0", lw=0.9)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"height $T$")
ax.set_ylabel(r"$N(T)-N_{\rm sm}(T)\approx S(T)$")
ax.grid(alpha=0.3)
plt.savefig("counting-zeros_Sfunc.pdf", bbox_inches="tight")
