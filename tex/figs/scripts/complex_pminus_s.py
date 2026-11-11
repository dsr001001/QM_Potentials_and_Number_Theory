import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
phi = np.linspace(0, 2 * np.pi, 400)
td = np.arange(0, 4.0001, 0.25)
for p, c in [(2, "C0"), (3, "C1"), (5, "C2")]:
    r = p ** -0.5
    a1.plot(r * np.cos(phi), r * np.sin(phi), color=c, lw=0.8)
    zz = p ** (-0.5 - 1j * td)
    a1.plot(zz.real, zz.imag, "o", color=c, ms=3, label=r"$p=%d$" % p)
a1.axhline(0, color="k", lw=0.5); a1.axvline(0, color="k", lw=0.5)
a1.set_aspect("equal"); a1.set_xlabel("real part"); a1.set_ylabel("imaginary part")
a1.legend(fontsize=7)
t = np.linspace(0, 30, 2000)
for p, c in [(2, "C0"), (3, "C1"), (5, "C2")]:
    a2.plot(t, -p ** -0.5 * np.sin(t * np.log(p)), color=c, label=r"$p=%d$" % p)
P2 = 2 * np.pi / np.log(2)
a2.annotate("", xy=(P2, 0.8), xytext=(0, 0.8), arrowprops=dict(arrowstyle="<->"))
a2.text(P2 / 2 - 2, 0.85, "9.065")
a2.set_ylim(-0.8, 1.0)
a2.set_xlabel(r"height $t$ (dimensionless)"); a2.set_ylabel(r"$\mathrm{Im}\,p^{-1/2-it}$")
a2.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("complex_pminus_s.pdf", bbox_inches="tight")
