import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath

t1 = float(mpmath.zetazero(1).imag)
s = 0.5 + 1j * t1
N = 300
k = np.arange(1, N + 2)
terms = (-1.0)**(k + 1) * np.exp(-s * np.log(k))
S = np.cumsum(terms)
avg = 0.5 * (S[:-1] + S[1:])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(S.real, S.imag, "-", lw=0.6, color="0.6", label=r"partial sums $S_N$")
ax.plot(S.real[:12], S.imag[:12], "o", ms=3, color="C0", label=r"$S_1,\ldots,S_{12}$")
ax.plot(avg.real, avg.imag, "-", color="C3", lw=1.2, label=r"averages $(S_N+S_{N+1})/2$")
ax.plot(0, 0, "k+", ms=14, mew=2)
ax.set_aspect("equal")
ax.set_xlabel(r"real part (dimensionless)")
ax.set_ylabel(r"imaginary part (dimensionless)")
ax.legend(fontsize=7, loc="best")
plt.savefig("zeta_eta.pdf", bbox_inches="tight")
