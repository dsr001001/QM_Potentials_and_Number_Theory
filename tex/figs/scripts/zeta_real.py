import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath

s1 = np.linspace(-1.0, 6.0, 1401)
z1 = np.array([float(mpmath.zeta(v)) if abs(v - 1.0) > 0.02 else np.nan for v in s1])
s2 = np.linspace(-8.5, -1.5, 701)
z2 = np.array([float(mpmath.zeta(v)) for v in s2])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(s1, z1, "k", lw=1.5, label=r"$\zeta(s)$")
sp = s1[s1 > 1.05]
a1.plot(sp, 1.0 / (sp - 1.0), ":", color="C0", label=r"$1/(s-1)$")
a1.axvline(1.0, color="C3", lw=0.8, ls="--")
a1.axhline(0, color="0.5", lw=0.5)
a1.plot([0, -1], [-0.5, -1.0 / 12], "o", color="C2", ms=4)
a1.set_ylim(-4, 8)
a1.set_xlabel(r"$s$ (real, dimensionless)")
a1.set_ylabel(r"$\zeta(s)$")
a1.set_title("(a)", fontsize=9)
a1.legend(fontsize=7)

a2.plot(s2, z2, "k", lw=1.5)
a2.axhline(0, color="0.5", lw=0.5)
a2.plot([-2, -4, -6, -8], [0, 0, 0, 0], "s", color="C0", ms=5, label="trivial zeros")
a2.set_xlabel(r"$s$ (real, dimensionless)")
a2.set_ylabel(r"$\zeta(s)$")
a2.set_title("(b)", fontsize=9)
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("zeta_real.pdf", bbox_inches="tight")
