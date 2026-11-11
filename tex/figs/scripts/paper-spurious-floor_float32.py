import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L, N = np.pi/2, 800
h = 2*L/(N+1)
E, psi = eigh_tridiagonal(2/h**2*np.ones(N), -1/h**2*np.ones(N-1), select='i', select_range=(0,59))
psi /= np.sqrt((psi**2).sum(axis=0)*h)
off = ~np.eye(60, dtype=bool)
G64 = abs((psi.T@psi*h)[off])
p32 = psi.astype(np.float32)
G32 = abs(((p32.T@p32)*np.float32(h))[off]).astype(float)
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(np.log10(G64[G64>0]), bins=40, alpha=0.7, label='float64')
ax.hist(np.log10(G32[G32>0]), bins=40, alpha=0.7, label='float32')
ax.set_xlabel(r'$\log_{10}|\langle n|m\rangle|$, $n\neq m$ (dimensionless)')
ax.set_ylabel('count')
ax.legend()
plt.savefig('paper-spurious-floor_float32.pdf', bbox_inches='tight')
