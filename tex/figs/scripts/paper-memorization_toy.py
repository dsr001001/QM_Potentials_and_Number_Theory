import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
L, N, M = 44.883, 4000, 400
x = qmnt.grid(L, N); h = 2*L/(N+1)
V = np.array([qmnt.wu_sprung_V(abs(xi)) for xi in x])
E, psi = qmnt.fd_levels(V, L, k=500, return_vectors=True)
if psi.shape[0] != N:
    psi = psi.T
dens = psi**2/(np.sum(psi**2, axis=0)*h)
phi = np.cos(np.outer(x, np.arange(M))*np.pi/L)
A = h*dens.T @ phi
n = np.arange(1, 501)
s = qmnt.smooth_zeros(n); g = qmnt.mean_gap(n)
f = -qmnt.Nfl(s, [2])
U, sig, Wt = np.linalg.svd(A[:400], full_matrices=False)
filt = sig/(sig**2 + 1e-6*sig[0]**2)
c = Wt.T @ (filt*(U.T @ (f[:400]*g[:400])))
p = (A @ c)/g
fig, ax = plt.subplots(figsize=(6, 4))
sel = n >= 300
ax.plot(n[sel], f[sel], '.-', ms=3, lw=0.6, color='0.4', label='target')
ax.plot(n[sel], p[sel], '-', lw=2, color='C3', label='linearized fit')
ax.axvspan(400.5, 500.5, color='0.85')
ax.set_xlabel('level index $n$')
ax.set_ylabel('fluctuation (mean spacings)')
ax.legend(fontsize=8)
plt.savefig('paper-memorization_toy.pdf', bbox_inches='tight')
