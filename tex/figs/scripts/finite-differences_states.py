import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L, N = 1.0, 15
h = 2*L/(N+1)
x = -L + h*np.arange(1, N+1)
w, U = eigh_tridiagonal(2/h**2*np.ones(N), -np.ones(N-1)/h**2)
xx = np.linspace(-L, L, 400)
fig, ax = plt.subplots(figsize=(6, 4))
for i, k in enumerate([1, 2, 3, 15]):
    psi = U[:, k-1]/np.sqrt(h)
    cont = np.sin(k*np.pi*(xx+L)/(2*L))/np.sqrt(L)
    cj = np.sin(k*np.pi*(x+L)/(2*L))/np.sqrt(L)
    if np.dot(psi, cj) < 0: psi = -psi
    off = 3*i
    ax.plot(xx, cont+off, '-', color='gray', lw=1)
    ax.plot(x, psi+off, 'o', ms=4, label=f'k={k}')
ax.set_xlabel(r'$x$ (dimensionless)'); ax.set_ylabel(r'$\psi$ (offset)')
ax.legend(fontsize=8)
plt.savefig("finite-differences_states.pdf", bbox_inches="tight")
