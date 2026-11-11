import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
def lm_fit(c, resid, jac, lam=1.0, maxiter=60, accept=0.999):
    r = resid(c); cur = np.sqrt(np.mean(r**2))
    for it in range(maxiter):
        U, s, Vt = np.linalg.svd(jac(c), full_matrices=False)
        Utr = U.T @ r
        accepted = False
        for lam_try in (0.2*lam, lam, 5*lam, 25*lam):
            f = s/(s**2 + lam_try*s[0]**2)
            c_try = c + Vt.T @ (f*Utr)
            r_try = resid(c_try); rms_try = np.sqrt(np.mean(r_try**2))
            if rms_try < accept*cur:
                c, r, cur = c_try, r_try, rms_try
                lam = max(0.5*lam_try, 1e-7)
                accepted = True
                break
        if not accepted:
            lam = 30*lam
            if lam > 1e4:
                break
    return c, cur
L = 10.0; N = 300; h = 2*L/(N+1); x = -L + h*np.arange(1, N+1); nfit = 30
E0 = eigh_tridiagonal(2/h**2*np.ones(N), -np.ones(N-1)/h**2, select='i', select_range=(0, nfit), eigvals_only=True)
gap = np.gradient(E0)[:nfit]
rng = np.random.default_rng(0); d = rng.standard_normal(nfit); d *= 0.2/np.sqrt(np.mean(d**2))
tgt = E0[:nfit] + d*gap
def toy_rms(M):
    Ph = np.cos(np.outer(x, np.arange(M))*np.pi/L)
    def resid(c):
        w = eigh_tridiagonal(2/h**2 + Ph @ c, -np.ones(N-1)/h**2, select='i', select_range=(0, nfit-1), eigvals_only=True)
        return (tgt - w)/gap
    def jac(c):
        w, vv = eigh_tridiagonal(2/h**2 + Ph @ c, -np.ones(N-1)/h**2, select='i', select_range=(0, nfit-1))
        return ((vv**2).T @ Ph)/gap[:, None]
    return lm_fit(np.zeros(M), resid, jac)[1]
Ms = [2, 5, 10, 15, 20, 25, 28, 30, 32, 35, 40]
rs = [max(toy_rms(M), 1e-16) for M in Ms]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(Ms, rs, 'bo-')
ax.axvline(nfit, color='gray', ls=':', label="M = number of fitted levels")
ax.set_xlabel("basis size M")
ax.set_ylabel("in-sample rms (local mean spacings)")
ax.legend(fontsize=8)
plt.savefig("fitting_toyfit.pdf", bbox_inches="tight")
