import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
V0 = 3.10073*np.pi
def Nsm(t): u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def xws(V):
    a = np.sqrt(V-V0); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b+a)/(b-a)))/np.pi
aa = np.linspace(0, np.sqrt(1200-V0), 400001); VV = V0 + aa**2
XX = np.concatenate([[0.0], xws(VV[1:])])
L = xws(1000.0)
def levels(N, k=500):
    x = -L + 2*L/(N+1)*np.arange(1, N+1); h = 2*L/(N+1)
    V = np.interp(np.abs(x), XX, VV)
    return eigh_tridiagonal(2/h**2+V, -np.ones(N-1)/h**2, select='i',
                            select_range=(0, k-1), eigvals_only=True)
ER = (4*levels(23999) - levels(11999))/3
n = np.arange(1, 501)
s = np.array([brentq(lambda t: Nsm(t)-(k-0.5), 7.0, 2000.0) for k in n])
g = 2*np.pi/np.log(s/(2*np.pi))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, (ER-s)/g, ".", ms=3, label="finite differences + Richardson")
ax.axhline(Nsm(V0), color="r", label="Bohr--Sommerfeld offset $N_{\\rm sm}(V_0)$")
ax.set_xlabel("level index $n$"); ax.set_ylabel("$(E_n-s_n)/\\mathrm{gap}_n$ (mean spacings)")
ax.legend(fontsize=8)
plt.savefig("wu-sprung_levels.pdf", bbox_inches="tight")
