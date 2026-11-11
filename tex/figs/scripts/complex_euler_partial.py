import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import factorial
S = np.cumsum([(1j * np.pi) ** n / factorial(n) for n in range(15)])
path = np.concatenate([[0], S])
fig, ax = plt.subplots(figsize=(6, 4))
phi = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(phi), np.sin(phi), "k--", lw=0.8, label=r"unit circle $|z|=1$")
ax.plot(path.real, path.imag, "o-", ms=3, color="C0", label=r"partial sums $S_N$")
for n in range(5):
    ax.annotate(r"$S_{%d}$" % n, (S[n].real, S[n].imag), textcoords="offset points", xytext=(5, 5))
ax.plot(-1, 0, "r*", ms=14, label=r"$e^{i\pi}=-1$")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_aspect("equal")
ax.set_xlabel("real part (dimensionless)"); ax.set_ylabel("imaginary part (dimensionless)")
ax.legend(loc="lower right", fontsize=8)
plt.savefig("complex_euler_partial.pdf", bbox_inches="tight")
