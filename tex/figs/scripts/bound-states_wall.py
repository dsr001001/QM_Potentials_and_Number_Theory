import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
h = 0.002
def E1(L):
    N = int(round(2*L/h)) - 1
    hh = 2*L/(N + 1)
    x = -L + hh*np.arange(1, N + 1)
    return eigh_tridiagonal(2/hh**2 + x**2, -np.ones(N - 1)/hh**2, eigvals_only=True,
                            select='i', select_range=(0, 0))[0]
Eref = E1(8.0)
Ls = np.arange(1.5, 3.76, 0.25)
shift = np.array([E1(L) - Eref for L in Ls])
Lf = np.linspace(1.4, 3.9, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(Ls, shift, 'o', label='numerical (tridiagonal)')
ax.semilogy(Lf, 4*Lf/np.sqrt(np.pi)*np.exp(-Lf**2), '-', label=r'estimate $(4L/\sqrt{\pi})e^{-L^2}$')
ax.set_xlabel(r"wall position $L$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"$E_b-1$ (units $\hbar=2m=1$)")
ax.legend()
plt.savefig("bound-states_wall.pdf", bbox_inches="tight")
