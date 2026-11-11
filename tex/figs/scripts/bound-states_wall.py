import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

def ground(L, N=6000):
    h = 2*L/(N + 1)
    x = -L + h*np.arange(1, N + 1)
    d = 2/h**2 + x**2
    e = -np.ones(N - 1)/h**2
    return eigh_tridiagonal(d, e, eigvals_only=True, select="i",
                            select_range=(0, 0))[0]

Ls = np.linspace(1.2, 3.5, 12)
shift = np.array([ground(L) - 1.0 for L in Ls])
Lf = np.linspace(1.2, 3.6, 200)
est = 4*Lf/np.sqrt(np.pi)*np.exp(-Lf**2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(Ls, shift, "o", label="numerical $E_b-1$")
ax.semilogy(Lf, est, "-", label=r"estimate $(4L/\sqrt{\pi})e^{-L^2}$")
ax.set_xlabel(r"wall position $L$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"$E_b-1$ (units $\hbar=2m=1$)")
ax.set_title("Oscillator ground state: effect of walls at $\\pm L$")
ax.legend()
plt.savefig("bound-states_wall.pdf", bbox_inches="tight")
