import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(0, 1, color="0.9", label=r"critical strip $0<\sigma<1$")
ax.axvline(0.5, color="k", ls="--", lw=0.8, label=r"critical line $\sigma=1/2$")
ax.plot(0.5 * np.ones(10), t, "o", color="C0", label=r"$\rho_n=1/2+it_n$")
ax.plot(0.5 * np.ones(10), -t, "o", mfc="none", color="C0", label=r"$\bar\rho_n$")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-1, 2); ax.set_ylim(-55, 55)
ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$ (dimensionless)"); ax.set_ylabel(r"$t=\mathrm{Im}\,s$ (dimensionless)")
ax.legend(fontsize=7, loc="center right")
plt.savefig("complex_critical_line.pdf", bbox_inches="tight")
