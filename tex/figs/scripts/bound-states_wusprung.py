import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
from scipy.linalg import eigh_tridiagonal
L = qmnt.L_of(1000)
xx = np.linspace(-L, L, 801)
VV = np.array([qmnt.wu_sprung_V(v) for v in xx])
t500 = 811.18
xt = qmnt.wu_sprung_x(t500)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(xx, VV, 'k')
a1.plot([-L, -L], [0, 1050], 'k', lw=4); a1.plot([L, L], [0, 1050], 'k', lw=4)
a1.axhline(t500, ls='--', color='gray')
a1.plot([-xt, xt], [t500, t500], 'ro', ms=4)
a1.set_xlabel(r'$x$ (units $\hbar=2m=1$)'); a1.set_ylabel(r'$V(x)$')
N = 3000
x = qmnt.grid(L, N)
h = x[1] - x[0]
V = np.array([qmnt.wu_sprung_V(v) for v in x])
w, vec = eigh_tridiagonal(2/h**2 + V, -np.ones(N - 1)/h**2, select='i', select_range=(0, 3))
xm = 1.6*qmnt.wu_sprung_x(w[3])
sel = np.abs(x) < xm
a2.plot(x[sel], V[sel], 'k')
for j in range(4):
    psi = vec[:, j]/np.max(np.abs(vec[:, j]))
    a2.axhline(w[j], color='gray', lw=0.5)
    a2.plot(x[sel], w[j] + 2.0*psi[sel], lw=1.3)
a2.set_ylim(qmnt.V0 - 1, w[3] + 6)
a2.set_xlabel(r'$x$ (units $\hbar=2m=1$)'); a2.set_ylabel('energy; states on baselines')
plt.tight_layout()
plt.savefig("bound-states_wusprung.pdf", bbox_inches="tight")
