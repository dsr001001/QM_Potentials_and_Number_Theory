import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath as mp
import qmnt
t1 = float(qmnt.zeros(1)[0])
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
# left: sigma = 2, converges
s = 2.0 + 1j * t1
k = np.arange(1, 201)
S = np.cumsum(np.exp(-s * np.log(k)))
z = complex(mp.zeta(mp.mpc(2.0, t1)))
ax = axs[0]
ax.plot(S.real, S.imag, "-o", ms=2, lw=0.8)
ax.plot(z.real, z.imag, "r*", ms=12, label=r"$\zeta(2+it_1)$")
ax.set_title(r"$s=2+i\,t_1$: converges", fontsize=9)
ax.set_xlabel(r"Re $S_n$"); ax.set_ylabel(r"Im $S_n$")
ax.legend(fontsize=8)
ax.set_aspect("equal", adjustable="datalim")
# right: sigma = 1/2, the first zero, diverges
s = 0.5 + 1j * t1
k = np.arange(1, 2001)
S = np.cumsum(np.exp(-s * np.log(k)))
nn = np.linspace(3, 2000, 4000)
main = np.exp((1 - s) * np.log(nn)) / (1 - s)
ax = axs[1]
ax.plot(S.real, S.imag, ".", ms=1.5, label=r"$S_n$")
ax.plot(main.real, main.imag, "k-", lw=0.5, label=r"$n^{1-s}/(1-s)$")
ax.plot(0, 0, "r+", ms=10)
ax.set_title(r"$s=\frac{1}{2}+i\,t_1$: spirals outward", fontsize=9)
ax.set_xlabel(r"Re $S_n$")
ax.legend(fontsize=7, loc="upper right")
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig("series_spiral.pdf", bbox_inches="tight")
