import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = 0.3
eps = np.linspace(-2, 2, 201)
lo, hi, olo, ohi = [], [], [], []
for e in eps:
    w, U = np.linalg.eigh(np.array([[e, g], [g, -e]]))
    lo.append(w[0]); hi.append(w[1])
    olo.append(abs(U[0, 0])**2); ohi.append(abs(U[0, 1])**2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
sc = a1.scatter(eps, lo, c=olo, cmap='coolwarm', vmin=0, vmax=1, s=6)
a1.scatter(eps, hi, c=ohi, cmap='coolwarm', vmin=0, vmax=1, s=6)
a1.plot(eps, eps, 'k--', lw=0.7); a1.plot(eps, -eps, 'k--', lw=0.7)
fig.colorbar(sc, ax=a1, label=r'overlap $|e_1^\dagger u|^2$')
a1.set_xlabel(r'$\varepsilon$ (dimensionless)'); a1.set_ylabel('eigenvalue (dimensionless)')
gg = np.linspace(0, 2, 401)
root = np.sqrt((1 - gg**2).astype(complex))
for sgn in [1, -1]:
    lam = 2 + sgn*root
    a2.plot(gg, lam.real, 'C0-', lw=1.5)
    a2.plot(gg, lam.imag, 'C1--', lw=1.5)
a2.plot([], [], 'C0-', label='real part'); a2.plot([], [], 'C1--', label='imaginary part')
a2.axvline(1, color='0.6', lw=0.6)
a2.set_xlabel('$g$ (dimensionless)'); a2.set_ylabel('eigenvalue (dimensionless)')
a2.legend(fontsize=8)
fig.tight_layout()
plt.savefig('eigen_avoided.pdf', bbox_inches='tight')
