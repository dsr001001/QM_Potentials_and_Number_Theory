import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
L = 10.0; N = 300; h = 2*L/(N+1); x = -L + h*np.arange(1, N+1); nf = 30
def solve(V, vec=False):
    d = 2/h**2 + V; e = -np.ones(N-1)/h**2
    return eigh_tridiagonal(d, e, eigvals_only=not vec, select='i', select_range=(0, nf-1))
E0 = solve(np.zeros(N)); gap = np.gradient(E0)
rng = np.random.default_rng(0); target = E0 + 0.2*gap*rng.standard_normal(nf)
Ms = [5, 10, 15, 20, 25, 30, 32, 35, 40]; out = []
for M in Ms:
    phi = np.cos(np.outer(np.arange(M), np.pi*x/L))
    c = np.zeros(M); res = lambda c: (target - solve(phi.T@c))/gap
    r = res(c); cur = np.sqrt(np.mean(r**2)); lam = 1.0
    for it in range(60):
        E, v = solve(phi.T@c, True); J = (v**2).T@phi.T/gap[:, None]
        U, s, Vt = np.linalg.svd(J, full_matrices=False); Ur = U.T@r; ok = False
        for lt in (0.2*lam, lam, 5*lam, 25*lam):
            ct = c + Vt.T@(s/(s**2 + lt*s[0]**2)*Ur); rt = res(ct); rm = np.sqrt(np.mean(rt**2))
            if rm < 0.999*cur:
                c, r, cur, lam, ok = ct, rt, rm, max(0.5*lt, 1e-7), True; break
        if not ok:
            lam *= 30
            if lam > 1e4: break
    out.append(max(cur, 1e-8))
plt.figure(figsize=(6, 4)); plt.semilogy(Ms, out, "o-")
plt.axvline(nf + 1, ls=":", c="gray")
plt.xlabel("basis size M"); plt.ylabel("in-sample rms (local spacings)")
plt.savefig("fitting_toyfit.pdf", bbox_inches="tight")
