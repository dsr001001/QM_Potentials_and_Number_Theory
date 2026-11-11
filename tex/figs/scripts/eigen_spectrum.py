import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L = 44.883; N = 600; h = 2*L/(N+1)
w = eigh_tridiagonal(2/h**2*np.ones(N), -np.ones(N-1)/h**2, eigvals_only=True)
k = np.arange(1, N+1); ex = (k*np.pi/(2*L))**2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(k, w, '.', ms=2, label='matrix $D_N/h^2$')
a1.plot(k, ex, '-', lw=1, label=r'exact $(k\pi/2L)^2$')
a1.set_xlabel('level index $k$'); a1.set_ylabel(r'eigenvalue ($\hbar=2m=1$)')
a1.legend(fontsize=8)
a2.plot(k, w/ex, '-')
a2.axhline(4/np.pi**2, color='0.5', ls='--', lw=0.8)
a2.set_xlabel('level index $k$'); a2.set_ylabel('matrix / exact')
fig.tight_layout()
plt.savefig('eigen_spectrum.pdf', bbox_inches='tight')
