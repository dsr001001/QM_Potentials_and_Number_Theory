import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
V0 = 3.10073*np.pi
def x_of_V(V):
    s, r = np.sqrt(V-V0), np.sqrt(V)
    return (s*np.log(V0/(2*np.pi*np.e**2))+r*np.log((r+s)**2/V0))/np.pi
u = np.linspace(0, np.sqrt(1100-V0), 40001)
sp = CubicSpline(x_of_V(V0+u**2), u)
L = x_of_V(1000.0)
def levels(N, k=500):
    h = 2*L/(N+1); x = -L+h*np.arange(1, N+1)
    V = V0+sp(np.abs(x))**2
    return eigh_tridiagonal(2/h**2+V, -np.ones(N-1)/h**2, eigvals_only=True,
                            select='i', select_range=(0, k-1))
Eg = levels(6000); Ec = levels(12000); Ef = levels(24000)
ER = (4*Ef-Ec)/3
Nsm = lambda t: t/(2*np.pi)*np.log(t/(2*np.pi))-t/(2*np.pi)+7/8
n = np.arange(1, 501)
s = np.array([brentq(lambda t: Nsm(t)-(k-0.5), 2*np.pi, 2000.0) for k in n])
gap = 2*np.pi/np.log(s/(2*np.pi))
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5,3))
a.plot(n, (Eg-ER)/gap, 'k-')
a.axvspan(401, 500, color='0.85')
a.set_xlabel('level n'); a.set_ylabel(r'$\Delta_n$ (mean spacings)')
b.plot(n, (ER-s)/gap, 'k.', ms=2)
b.set_xlabel('level n'); b.set_ylabel(r'$(E^{Rich}_n-s_n)$ (mean spacings)')
plt.tight_layout()
plt.savefig("finite-differences_wsbias.pdf", bbox_inches="tight")
