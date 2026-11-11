import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath as mp
t1 = 14.134725141734695
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
s = 2.0 + 1j*t1
k = np.arange(1, 201).astype(float)
S = np.cumsum(np.exp(-s*np.log(k)))
z = complex(mp.zeta(mp.mpc(2.0, t1)))
a1.plot(S.real, S.imag, '.-', ms=2, lw=0.6)
a1.plot(z.real, z.imag, 'r*', ms=12, label=r'$\zeta(2+it_1)$')
a1.set_title(r'$s=2+it_1$', fontsize=9)
a1.legend(fontsize=8)
s = 0.5 + 1j*t1
k = np.arange(1, 1001).astype(float)
S = np.cumsum(np.exp(-s*np.log(k)))
main = k**(1 - s)/(1 - s)
a2.plot(S.real, S.imag, lw=0.6, label='$S_n$')
a2.plot(main.real, main.imag, 'k', lw=0.4, label=r'$n^{1-s}/(1-s)$')
a2.plot(0, 0, 'r+', ms=10)
a2.set_title(r'$s=\frac{1}{2}+it_1$ (first zero)', fontsize=9)
a2.legend(fontsize=8)
for ax in (a1, a2):
    ax.set_xlabel(r'$\mathrm{Re}\,S_n$ (dimensionless)')
    ax.set_ylabel(r'$\mathrm{Im}\,S_n$ (dimensionless)')
    ax.set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.savefig("series_spiral.pdf", bbox_inches="tight")
