import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

V0 = 3.10073*np.pi
def x_of_V(V):
    a = np.sqrt(max(V - V0, 0.0)); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b + a)/(b - a)))/np.pi
def V_of_x(x):
    x = abs(x)
    if x == 0.0:
        return V0
    return brentq(lambda V: x_of_V(V) - x, V0, 5000.0, xtol=1e-12)

def Nsm(t):
    return t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8

L, N = 44.883, 3000
h = 2*L/(N + 1)
x = -L + h*np.arange(1, N + 1)
half = np.array([V_of_x(xi) for xi in x[N//2:]])
V = np.concatenate([half[::-1], half])
E, psi = eigh_tridiagonal(2/h**2 + V, -np.ones(N - 1)/h**2,
                          select='i', select_range=(0, 5))
psi = psi/np.sqrt(np.sum(psi**2, axis=0)*h)
s = [brentq(lambda t: Nsm(t) - (n - 0.5), 2*np.pi, 1e4) for n in range(1, 7)]

fig, ax = plt.subplots(figsize=(6, 4))
m = np.abs(x) < 8
ax.plot(x[m], V[m], "k-", lw=2, label="V(x)")
for n in range(6):
    ax.axhline(E[n], color="gray", lw=0.5)
    ax.plot(x[m], E[n] + 3.0*psi[m, n], lw=1)
    ax.plot([7.3, 8.0], [s[n], s[n]], "r-", lw=1.5)
ax.set_xlim(-8, 8); ax.set_ylim(0, 60)
ax.set_xlabel(r"position $x$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"energy, $V$ (units $\hbar=2m=1$)")
ax.legend(fontsize=8, loc="upper center")
plt.savefig("schrodinger_wusprung.pdf", bbox_inches="tight")
