import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lx = lp = np.sqrt(2*np.pi)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0.3, 20, 800)
for E, c in [(20, "C0"), (40, "C1"), (80, "C2")]:
    ax.plot(x, E/x, color=c, lw=1.2, label=f"$xp={E}$")
E = 40.0
xs = np.linspace(lx, E/lp, 400)
ax.fill_between(xs, lp, E/xs, color="C1", alpha=0.25, label=r"counted region, $E=40$")
ax.axvline(lx, color="k", ls="--", lw=0.8)
ax.axhline(lp, color="k", ls="--", lw=0.8)
for x0 in [3.0, 6.0, 10.0]:
    ax.annotate("", xy=(x0*1.25, 40/(x0*1.25)), xytext=(x0, 40/x0),
                arrowprops=dict(arrowstyle="->", color="C1", lw=1.2))
ax.text(lx + 0.2, 18, r"$x=\ell_x$", fontsize=8)
ax.text(14, lp + 0.3, r"$p=\ell_p$", fontsize=8)
ax.set_xlim(0, 20); ax.set_ylim(0, 20)
ax.set_xlabel("position $x$ (units with $\\hbar=1$)")
ax.set_ylabel("momentum $p$ (units with $\\hbar=1$)")
ax.legend(fontsize=8, loc="upper right")
plt.savefig("hilbert-polya_xp.pdf", bbox_inches="tight")
