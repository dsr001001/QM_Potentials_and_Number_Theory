import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
V0 = 3.10073*np.pi
def xV(V):
    V = np.asarray(V, float); a = np.sqrt(np.maximum(V - V0, 0.0)); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b + a)/(b - a)))/np.pi
def Nsm(t):
    u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def szero(n):
    return brentq(lambda t: Nsm(t) - (n - 0.5), 7.0, 1e6)
L = float(xV(1000.0))
u = np.linspace(0, np.sqrt(1200 - V0), 40001); Vg = V0 + u**2; xg = xV(Vg)
N = 4000; h = 2*L/(N + 1); x = -L + h*np.arange(1, N + 1)
Vx = np.interp(np.abs(x), xg, Vg)
E, psi = eigh_tridiagonal(2/h**2 + Vx, -np.ones(N - 1)/h**2, select="i", select_range=(0, 499))
dens = psi**2/h                      # normalized so that sum(dens)*h = 1
M = 400
phi = np.cos(np.outer(x, np.arange(M))*np.pi/L)
A = h*(dens.T @ phi)                 # A_nk, shape (500, M)
n = np.arange(1, 501)
s = np.array([szero(k) for k in n]); g = 2*np.pi/np.log(s/(2*np.pi))
re2 = np.zeros_like(s)
for m in (1, 2, 3):
    re2 += np.sin(m*s*np.log(2))/(m*2**(m/2))
f = re2/np.pi                        # target fluctuation -N_fl(s_n), mean spacings
y = f*g                              # energy units
U, S, Vt = np.linalg.svd(A[:400], full_matrices=False)
filt = S/(S**2 + 1e-6*S[0]**2)
c = Vt.T @ (filt*(U.T @ y[:400]))
pred = (A @ c)/g
fig, ax = plt.subplots(figsize=(6, 4))
sel = n >= 300
ax.plot(n[sel], f[sel], ".-", color="0.4", lw=0.7, ms=3, label="target")
ax.plot(n[sel], pred[sel], "-", color="C3", lw=1.6, label="linearized fit, one step")
ax.axvspan(400.5, 500.5, color="0.88")
ax.set_xlabel("level index n")
ax.set_ylabel("fluctuation (mean spacings)")
ax.legend(fontsize=8, loc="upper left")
plt.savefig("paper-memorization_toy.pdf", bbox_inches="tight")
