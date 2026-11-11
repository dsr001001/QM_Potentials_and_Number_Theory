import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L, N = 1.0, 7
h = 2*L/(N+1)
xall = -L + h*np.arange(N+2)
xin = xall[1:-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
xx = np.linspace(-L, L, 300)
a1.plot(xx, np.cos(np.pi*xx/2), '-', color='gray', lw=1, label='continuum ground state')
a1.plot(xin, np.cos(np.pi*xin/2), 'o', label='interior unknowns')
a1.plot([-L, L], [0, 0], 's', mfc='none', ms=8, label=r'walls, $\psi=0$')
a1.axhline(0, color='k', lw=0.5)
a1.set_xlabel(r'$x$ (dimensionless)'); a1.set_ylabel(r'$\psi$ (unnormalized)')
a1.legend(fontsize=7)
M = np.zeros((10, 10))
for i in range(10):
    M[i, i] = 2
    if i > 0: M[i, i-1] = -1
    if i < 9: M[i, i+1] = -1
a2.imshow(M != 0, cmap='Greys')
a2.set_title(r'nonzero pattern of $\mathsf{H}$ ($10\times10$)', fontsize=9)
a2.set_xlabel('column $j$'); a2.set_ylabel('row $j$')
plt.tight_layout()
plt.savefig("finite-differences_grid.pdf", bbox_inches="tight")
