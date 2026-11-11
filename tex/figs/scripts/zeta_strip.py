import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath

t = [float(mpmath.zetazero(n).imag) for n in range(1, 11)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(1, 3, color="#dde8f7", label=r"series converges absolutely ($\sigma>1$)")
ax.axvspan(0, 1, color="0.82", label=r"critical strip $0<\sigma<1$")
ax.axvline(0.5, color="k", ls="--", lw=1, label=r"critical line $\sigma=1/2$")
ax.axhline(0, color="0.5", lw=0.5)
ax.plot([0.5] * 10, t, "o", color="C3", ms=4, label="nontrivial zeros")
ax.plot([0.5] * 10, [-v for v in t], "o", color="C3", ms=4)
ax.plot([-2, -4, -6, -8], [0, 0, 0, 0], "s", color="C0", ms=5, label="trivial zeros")
ax.plot([1], [0], "x", color="k", ms=9, mew=2, label="pole at $s=1$")
ax.set_xlim(-9, 3)
ax.set_ylim(-55, 55)
ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$ (dimensionless)")
ax.set_ylabel(r"$t=\mathrm{Im}\,s$ (dimensionless)")
ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
plt.savefig("zeta_strip.pdf", bbox_inches="tight")
