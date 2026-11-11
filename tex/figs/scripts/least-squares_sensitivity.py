import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
L = qmnt.L_of(1000); N = 6000
x = qmnt.grid(L, N); h = 2*L/(N + 1)
V = np.array([qmnt.wu_sprung_V(xi) for xi in x])
E, psi = qmnt.fd_levels(V, L, k=400, return_vectors=True)
psi = np.asarray(psi)
if psi.shape[0] != N:
    psi = psi.T
psi = psi/np.sqrt(np.sum(psi**2, axis=0)*h)
fig, ax = plt.subplots(figsize=(6, 4))
for M, st in [(400, 'b-'), (900, 'r--')]:
    phi = np.cos(np.outer(x, np.arange(M))*np.pi/L)
    A = (psi**2*h).T @ phi
    s = np.linalg.svd(A, compute_uv=False)
    ax.semilogy(np.arange(1, len(s) + 1), s/s[0], st, label='M = %d' % M)
ax.set_xlabel('index j'); ax.set_ylabel('sigma_j / sigma_1 (dimensionless)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.savefig('least-squares_sensitivity.pdf', bbox_inches='tight')
