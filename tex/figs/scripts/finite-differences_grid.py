import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5,3))
N = 7; L = 1.0; h = 2*L/(N+1); x = -L+h*np.arange(0, N+2)
xx = np.linspace(-1, 1, 200)
a.plot(xx, np.cos(np.pi*xx/2), 'k-', lw=0.8)
a.plot(x[1:-1], np.cos(np.pi*x[1:-1]/2), 'o', color='C0', label='unknowns')
a.plot([x[0], x[-1]], [0, 0], 's', mfc='none', color='C3', ms=8, label='walls')
a.set_xlabel('x (dimensionless)'); a.set_ylabel(r'$\psi$ (unnormalized)')
a.legend(fontsize=8)
M = 10
H = np.diag(2*np.ones(M))-np.diag(np.ones(M-1), 1)-np.diag(np.ones(M-1), -1)
b.spy(H, markersize=8)
b.set_title('nonzero entries of H', fontsize=10)
plt.savefig("finite-differences_grid.pdf", bbox_inches="tight")
