import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import qmnt
L = 44.883; N = 4000; h = 2*L/(N+1)
x = -L + h*np.arange(1, N+1)
V = np.array([float(qmnt.wu_sprung_V(float(xi))) for xi in x])
E, psi = eigh_tridiagonal(2/h**2 + V, -np.ones(N-1)/h**2, select='i', select_range=(0, 19))
psi = psi/np.sqrt(h)
m = np.abs(x) <= 8
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x[m], V[m], 'k-', lw=1.2, label='Wu--Sprung $V(x)$')
for n in [1, 2, 3, 4, 20]:
    ax.axhline(E[n-1], color='0.8', lw=0.5)
    ax.plot(x[m], E[n-1] + 6*psi[m, n-1], lw=1, label='n = %d' % n)
ax.set_ylim(0, E[19] + 10)
ax.set_xlabel(r'$x$ (units $\hbar=2m=1$)')
ax.set_ylabel('energy (units $\hbar=2m=1$)')
ax.legend(fontsize=7, loc='upper center', ncol=3)
plt.savefig('eigen_wavefunctions.pdf', bbox_inches='tight')
