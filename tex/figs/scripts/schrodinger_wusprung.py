import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
V0 = 3.10073*np.pi
def x_of_V(V):
    a = np.sqrt(max(V - V0, 0.0)); b = np.sqrt(V)
    if a == 0.0:
        return 0.0
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b + a)/(b - a)))/np.pi
def V_of_x(x):
    x = abs(x)
    if x < 1e-14:
        return V0
    return brentq(lambda V: x_of_V(V) - x, V0, 5000.0, xtol=1e-12)
L, N = 44.883, 6000
h = 2*L/(N + 1)
x = -L + h*np.arange(1, N + 1)
half = np.array([V_of_x(xi) for xi in x[N//2:]])
V = np.concatenate([half[::-1], half])
E, P = eigh_tridiagonal(2/h**2 + V, -np.ones(N - 1)/h**2, select='i', select_range=(0, 5))
P = P/np.sqrt(np.sum(P**2, axis=0)*h)
Nsm = lambda t: t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
s = [brentq(lambda t: Nsm(t) - (n - 0.5), 2*np.pi, 1e4) for n in range(1, 7)]
fig, ax = plt.subplots(figsize=(6, 4))
m = np.abs(x) < 4
ax.plot(x[m], V[m], "k", lw=2, label="V(x)")
for n in range(6):
    ax.axhline(E[n], color="0.7", lw=0.5)
    ax.plot(x[m], E[n] + 2.5*P[m, n], lw=1)
    ax.plot([3.8, 4.0], [s[n], s[n]], "r", lw=1.5)
ax.set_xlim(-4, 4); ax.set_ylim(0, 50)
ax.set_xlabel("x (units hbar = 2m = 1)")
ax.set_ylabel("V, E (units hbar = 2m = 1)")
ax.legend(loc="upper center")
plt.savefig("schrodinger_wusprung.pdf", bbox_inches="tight")
