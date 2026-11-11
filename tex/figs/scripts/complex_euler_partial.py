import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import factorial

theta = np.pi
S = []
acc = 0j
for n in range(15):
    acc = acc + (1j * theta) ** n / factorial(n)
    S.append(acc)
S = np.array(S)
fig, ax = plt.subplots(figsize=(6, 4))
phi = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(phi), np.sin(phi), "k--", lw=0.8, label="unit circle")
ax.plot(S.real, S.imag, "o-", color="C0", ms=4, label=r"partial sums $S_N$")
for n in range(7):
    ax.text(S[n].real + 0.12, S[n].imag + 0.12, rf"$S_{n}$", fontsize=9)
ax.plot(-1, 0, "r*", ms=14, label=r"$e^{i\pi}=-1$")
ax.axhline(0, color="gray", lw=0.6)
ax.axvline(0, color="gray", lw=0.6)
ax.set_aspect("equal")
ax.set_xlim(-4.8, 2.0)
ax.set_ylim(-2.6, 3.7)
ax.set_xlabel(r"$\mathrm{Re}\,S_N$ (dimensionless)")
ax.set_ylabel(r"$\mathrm{Im}\,S_N$ (dimensionless)")
ax.legend(loc="upper right", fontsize=8)
plt.savefig("complex_euler_partial.pdf", bbox_inches="tight")
