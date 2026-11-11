import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
N = 20
w, V = eigh_tridiagonal(2*np.ones(N), -np.ones(N-1))
j = np.arange(1, N+1); s = j/(N+1); sf = np.linspace(0, 1, 800)
fig, ax = plt.subplots(figsize=(6, 4))
for k, col in zip([1, 2, 3, 20], ['C0', 'C1', 'C2', 'C3']):
    v = V[:, k-1]; ref = np.sin(k*np.pi*s)
    v = v*(v @ ref)/(v @ v)
    ax.plot(s, v, 'o', color=col, ms=4, label='k = %d' % k)
    ax.plot(sf, np.sin(k*np.pi*sf), '-', color=col, lw=0.8, alpha=0.6 if k == 20 else 1.0)
ax.set_xlabel('position along chain, $s=j/21$ (dimensionless)')
ax.set_ylabel('eigenvector component (dimensionless)')
ax.legend(fontsize=8, loc='lower left')
plt.savefig('eigen_modes.pdf', bbox_inches='tight')
