import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(0, 1, color="0.9", label=r"critical strip $0<\sigma<1$")
ax.axvline(0.5, color="k", ls="--", lw=1, label=r"critical line $\sigma=1/2$")
ax.plot(0.5 + 0 * t, t, "o", color="C0", label=r"$\rho_n=\frac{1}{2}+it_n$")
ax.plot(0.5 + 0 * t, -t, "o", mfc="none", color="C0", label=r"$\bar\rho_n=\frac{1}{2}-it_n$")
ax.text(0.56, t[0] - 1.2, r"$\rho_1$", fontsize=9)
ax.axhline(0, color="gray", lw=0.6)
ax.axvline(0, color="gray", lw=0.6)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-56, 56)
ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$ (dimensionless)")
ax.set_ylabel(r"$t=\mathrm{Im}\,s$ (dimensionless)")
ax.legend(fontsize=7, loc="center right")
plt.savefig("complex_critical_line.pdf", bbox_inches="tight")
