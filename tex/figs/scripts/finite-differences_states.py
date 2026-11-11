import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
N = 15; L = 1.0; h = 2*L/(N+1); x = -L+h*np.arange(1, N+1)
H = (np.diag(2*np.ones(N))-np.diag(np.ones(N-1), 1)-np.diag(np.ones(N-1), -1))/h**2
w, U = np.linalg.eigh(H)
xx = np.linspace(-L, L, 400)
fig, ax = plt.subplots(figsize=(6,4))
for off, k in enumerate([1, 2, 3, 15]):
    cont = np.sin(k*np.pi*(xx+L)/(2*L))/np.sqrt(L)
    psi = U[:, k-1]/np.sqrt(h)
    ref = np.sin(k*np.pi*(x+L)/(2*L))
    psi = psi*np.sign(np.dot(psi, ref))
    ax.plot(xx, cont+3*off, 'k-', lw=0.8)
    ax.plot(x, psi+3*off, 'o', ms=4, color='C%d' % off, label='k=%d' % k)
ax.set_xlabel('x (dimensionless)'); ax.set_ylabel(r'$\psi$ (offset)')
ax.legend(fontsize=8)
plt.savefig("finite-differences_states.pdf", bbox_inches="tight")
