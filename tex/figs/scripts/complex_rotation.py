import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
phi = np.linspace(0, 2 * np.pi, 300)
a1.plot(np.cos(phi), np.sin(phi), "k-", lw=0.8)
th = 0.9
p = np.exp(1j * th)
a1.annotate("", xy=(p.real, p.imag), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="C0", lw=1.8))
a1.plot([p.real, p.real], [0, p.imag], "C0:", lw=1)
a1.plot([0, p.real], [p.imag, p.imag], "C0:", lw=1)
a1.text(p.real / 2 - 0.15, -0.2, r"$\cos\theta$", color="C0", fontsize=8)
a1.text(p.real + 0.05, p.imag / 2, r"$\sin\theta$", color="C0", fontsize=8)
a1.text(p.real + 0.05, p.imag + 0.05, r"$e^{i\theta}$", color="C0")
w = 1.5 + 0.4j
w2 = np.exp(1j * np.pi / 3) * w
for v, c, lab in [(w, "C1", r"$w$"), (w2, "C3", r"$e^{i\pi/3}w$")]:
    a1.annotate("", xy=(v.real, v.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.4, ls="--"))
    a1.text(v.real + 0.05, v.imag + 0.05, lab, color=c, fontsize=8)
a1.axhline(0, color="gray", lw=0.6)
a1.axvline(0, color="gray", lw=0.6)
a1.set_aspect("equal")
a1.set_xlim(-1.3, 2.0)
a1.set_ylim(-1.3, 1.9)
a1.set_xlabel(r"$\mathrm{Re}$ (dimensionless)")
a1.set_ylabel(r"$\mathrm{Im}$ (dimensionless)")
t = np.linspace(0, 4 * np.pi, 500)
a2.plot(t, np.real(np.exp(1j * t)), label=r"$\mathrm{Re}\,e^{i\theta}=\cos\theta$")
a2.plot(t, np.imag(np.exp(1j * t)), "--", label=r"$\mathrm{Im}\,e^{i\theta}=\sin\theta$")
a2.set_xlabel(r"angle $\theta$ (rad)")
a2.set_ylabel("value (dimensionless)")
a2.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig("complex_rotation.pdf", bbox_inches="tight")
