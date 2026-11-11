import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
phi = np.linspace(0, 2 * np.pi, 300)
ts = np.arange(0, 4.0001, 0.25)
for p, c in [(2, "C0"), (3, "C1"), (5, "C2")]:
    r = p ** -0.5
    a1.plot(r * np.cos(phi), r * np.sin(phi), color=c, lw=0.6)
    w = p ** (-0.5 - 1j * ts)
    a1.plot(w.real, w.imag, "o", color=c, ms=3, label=rf"$p={p}$")
    a1.plot(w[0].real, w[0].imag, "s", color=c, ms=6)
a1.axhline(0, color="gray", lw=0.6)
a1.axvline(0, color="gray", lw=0.6)
a1.set_aspect("equal")
a1.set_xlabel(r"$\mathrm{Re}\,p^{-s}$ (dimensionless)")
a1.set_ylabel(r"$\mathrm{Im}\,p^{-s}$ (dimensionless)")
a1.legend(fontsize=7, loc="upper right")
t = np.linspace(0, 30, 1500)
for p, c in [(2, "C0"), (3, "C1"), (5, "C2")]:
    a2.plot(t, -p ** -0.5 * np.sin(t * np.log(p)), color=c, lw=1, label=rf"$p={p}$")
P2 = 2 * np.pi / np.log(2)
a2.annotate("", xy=(P2, 0.8), xytext=(0, 0.8),
            arrowprops=dict(arrowstyle="<->", color="C0"))
a2.text(1.0, 0.85, r"$2\pi/\log 2$", color="C0", fontsize=8)
a2.set_ylim(-0.85, 1.0)
a2.set_xlabel(r"height $t$ (dimensionless)")
a2.set_ylabel(r"$\mathrm{Im}\,p^{-1/2-it}$")
a2.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("complex_pminus_s.pdf", bbox_inches="tight")
