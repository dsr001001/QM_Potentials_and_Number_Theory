import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath

t = np.linspace(0.0, 50.0, 1001)
z = np.array([complex(mpmath.zeta(0.5 + 1j * v)) for v in t])
Z = np.array([float(mpmath.siegelz(v)) for v in t])
tz = [float(mpmath.zetazero(n).imag) for n in range(1, 11)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for v in tz:
    a1.axvline(v, color="0.75", lw=0.6)
    a2.axvline(v, color="0.75", lw=0.6)
a1.plot(t, z.real, "k", lw=1.0, label=r"Re $\zeta(1/2+it)$")
a1.plot(t, z.imag, "--", color="C0", lw=1.0, label=r"Im $\zeta(1/2+it)$")
a1.axhline(0, color="0.5", lw=0.5)
a1.set_xlabel(r"$t$ (dimensionless)")
a1.set_ylabel("value (dimensionless)")
a1.set_title("(a)", fontsize=9)
a1.legend(fontsize=7)
a2.plot(t, Z, "k", lw=1.0)
a2.axhline(0, color="0.5", lw=0.5)
a2.set_xlabel(r"$t$ (dimensionless)")
a2.set_ylabel(r"$Z(t)$ (dimensionless)")
a2.set_title("(b)", fontsize=9)
plt.tight_layout()
plt.savefig("zeta_Z.pdf", bbox_inches="tight")
