import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
V0 = 3.10073*np.pi
def x_of_V(V):
    s, r = np.sqrt(V-V0), np.sqrt(V)
    return (s*np.log(V0/(2*np.pi*np.e**2)) + r*np.log((r+s)**2/V0))/np.pi
u = np.linspace(0, np.sqrt(1100-V0), 40001)
sp = CubicSpline(x_of_V(V0+u**2), u)
L = x_of_V(1000.0)
def lev(N, k=500):
    h = 2*L/(N+1); x = -L + h*np.arange(1, N+1)
    V = V0 + sp(np.abs(x))**2
    return eigh_tridiagonal(2/h**2+V, -np.ones(N-1)/h**2, eigvals_only=True,
                            select='i', select_range=(0, k-1))
Eg, E1, E2 = lev(6000), lev(12000), lev(24000)
r = 24001/12001
ER = (r**2*E2 - E1)/(r**2-1)
n = np.arange(1, 501)
g = qmnt.mean_gap(n); s = qmnt.smooth_zeros(n)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(n, (Eg-ER)/g)
a1.axvspan(401, 500, alpha=0.2, color='orange')
a1.set_xlabel('level $n$'); a1.set_ylabel(r'$\Delta_n/\mathrm{gap}_n$')
a2.plot(n, (ER-s)/g)
a2.set_xlabel('level $n$'); a2.set_ylabel(r'$(E^{\rm Rich}_n-s_n)/\mathrm{gap}_n$')
plt.tight_layout()
plt.savefig("finite-differences_wsbias.pdf", bbox_inches="tight")
