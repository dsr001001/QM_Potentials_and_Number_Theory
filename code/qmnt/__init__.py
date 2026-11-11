"""qmnt: companion library for "Potentials, Primes and Prediction".

Small, readable implementations of every computation the book needs:
zeta zeros, the smooth counting function, the truncated explicit formula,
the Wu--Sprung potential, a finite-difference Schrodinger solver, and the
paper's Hermitian potential fit (reduced cost).

Conventions follow the paper (D. Soni, revision R1, 2026): indices n start
at 1, units hbar = 2m = 1, errors in units of the local mean spacing.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import PchipInterpolator

_DATA = os.path.join(os.path.dirname(__file__), "data")
TWO_PI = 2.0 * np.pi

# ----------------------------------------------------------------- zeros
_ZEROS = None


def zeros(nmax: int) -> np.ndarray:
    """Imaginary parts t_1..t_nmax of the nontrivial zeta zeros (nmax <= 10000)."""
    global _ZEROS
    if _ZEROS is None:
        path = os.path.join(_DATA, "zeros.npy")
        if os.path.exists(path):
            _ZEROS = np.load(path)
        else:  # slow fallback, fine for small nmax
            import mpmath
            mpmath.mp.dps = 20
            _ZEROS = np.array([float(mpmath.zetazero(n).imag) for n in range(1, nmax + 1)])
    if nmax > len(_ZEROS):
        raise ValueError(f"only {len(_ZEROS)} zeros available")
    return _ZEROS[:nmax].copy()


# --------------------------------------------------- smooth counting function
def Nsm(t):
    """Smooth zero-counting function, paper Eq. (1)."""
    t = np.asarray(t, dtype=float)
    u = t / TWO_PI
    return u * np.log(u) - u + 7.0 / 8.0


def dNsm(t):
    """Derivative of Nsm: the mean density of zeros, log(t/2pi)/(2pi)."""
    return np.log(np.asarray(t, dtype=float) / TWO_PI) / TWO_PI


def smooth_zeros(n):
    """s_n solving Nsm(s_n) = n - 1/2 (closed form via Lambert W, then Newton polish)."""
    n = np.asarray(n, dtype=float)
    x = (n - 0.5 - 7.0 / 8.0) / np.e
    s = TWO_PI * np.e * np.exp(np.real(lambertw(x)))
    for _ in range(3):
        s = s - (Nsm(s) - (n - 0.5)) / dNsm(s)
    return s


def mean_gap(n):
    """gap_n = ds_n/dn = 2pi / log(s_n / 2pi)."""
    return 1.0 / dNsm(smooth_zeros(n))


def fluct(t, n):
    """Fluctuation (t - s_n)/gap_n in units of the local mean spacing."""
    return (np.asarray(t, dtype=float) - smooth_zeros(n)) / mean_gap(n)


# ----------------------------------------------------------------- primes
def primes_upto(x: int) -> np.ndarray:
    """All primes <= x (sieve of Eratosthenes)."""
    x = int(x)
    if x < 2:
        return np.array([], dtype=int)
    sieve = np.ones(x + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(x ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0]


def first_primes(k: int) -> np.ndarray:
    """The first k primes."""
    bound = 15
    while True:
        p = primes_upto(bound)
        if len(p) >= k:
            return p[:k]
        bound *= 2


# ------------------------------------------------------ explicit formula
def Nfl(t, primes, mmax: int = 3):
    """Truncated explicit-formula fluctuation, paper Eq. (4)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for p in np.atleast_1d(primes):
        lp = np.log(float(p))
        for m in range(1, mmax + 1):
            out -= np.sin(m * t * lp) / (m * float(p) ** (m / 2.0))
    return out / np.pi


def oracle_zeros(n, primes, mmax: int = 3):
    """Predicted zeros: solve Nsm(t) + Nfl(t) = n - 1/2 near s_n (App. A step 8)."""
    n = np.atleast_1d(np.asarray(n))
    s = smooth_zeros(n)
    g = mean_gap(n)
    out = np.empty(len(n))
    for i, (nn, ss, gg) in enumerate(zip(n, s, g)):
        f = lambda t: float(Nsm(t) + Nfl(t, primes, mmax)) - (nn - 0.5)
        w = 3.0
        a, b = max(ss - w * gg, 0.5), ss + w * gg
        while f(a) * f(b) > 0:          # bracket expanded on sign failure
            w *= 1.5
            a, b = max(ss - w * gg, 0.5), ss + w * gg
        out[i] = brentq(f, a, b, xtol=1e-12)
    return out


# ------------------------------------------------------------- scoring
def rms(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def corr(a, b) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def score(t_pred, n):
    """(rms, corr) of predicted vs true zeros with indices n, in mean spacings.

    rms: of (t_pred - t_n)/gap_n.  corr: between predicted and true
    fluctuations, both measured from the smooth zeros s_n."""
    n = np.asarray(n)
    t = zeros(int(n.max()))[n - 1]
    g = mean_gap(n)
    s = smooth_zeros(n)
    return rms((np.asarray(t_pred) - t) / g), corr((np.asarray(t_pred) - s) / g, (t - s) / g)


# ------------------------------------------------------ Wu--Sprung potential
V0 = 3.10073 * np.pi


def wu_sprung_x(V):
    """Paper Eq. (2): x(V) for V >= V0."""
    V = np.asarray(V, dtype=float)
    d = np.sqrt(np.maximum(V - V0, 0.0))
    sv = np.sqrt(V)
    return (d * np.log(V0 / (TWO_PI * np.e ** 2)) + sv * np.log((sv + d) / (sv - d))) / np.pi


def L_of(Vmax: float) -> float:
    """Half-width of the box: x(Vmax). 44.883 for 1000, 59.957 for 1500."""
    return float(wu_sprung_x(Vmax))


_VTAB = {}


def _vtable(Vtop):
    key = round(Vtop, 3)
    if key not in _VTAB:
        u = np.linspace(0.0, np.sqrt(Vtop - V0), 400001)  # V = V0 + u^2, dense
        V = V0 + u * u
        x = wu_sprung_x(V)
        _VTAB[key] = PchipInterpolator(x, V)
    return _VTAB[key]


def wu_sprung_V(x, Vtop: float = 2500.0):
    """Inverse of Eq. (2): the even Wu--Sprung potential V(x), V(0) = V0."""
    ax = np.abs(np.asarray(x, dtype=float))
    return _vtable(Vtop)(ax)


# ----------------------------------------------------- finite differences
def grid(L: float, N: int) -> np.ndarray:
    """N uniform interior points of (-L, L); spacing h = 2L/(N+1)."""
    h = 2.0 * L / (N + 1)
    return -L + h * np.arange(1, N + 1)


def fd_levels(Vvals, L: float, k: int | None = None, return_vectors: bool = False):
    """Eigenpairs of -d2/dx2 + V on the interior grid with Dirichlet walls.

    Vvals: potential at the N interior points. Returns the lowest k eigenvalues
    (all if k is None); eigenvectors normalized so that sum(psi^2) h = 1."""
    V = np.asarray(Vvals, dtype=float)
    N = len(V)
    h = 2.0 * L / (N + 1)
    d = 2.0 / h ** 2 + V
    e = -np.ones(N - 1) / h ** 2
    sel = "a" if k is None else "i"
    rng = None if k is None else (0, k - 1)
    if return_vectors:
        w, v = eigh_tridiagonal(d, e, select=sel, select_range=rng)
        return w, v / np.sqrt(h)
    return eigh_tridiagonal(d, e, eigvals_only=True, select=sel, select_range=rng)


def richardson(E_h, E_h2):
    """Richardson extrapolation over grid doubling: (4 E_{h/2} - E_h)/3."""
    return (4.0 * np.asarray(E_h2) - np.asarray(E_h)) / 3.0


# ----------------------------------------------------- Hermitian fit
SCALES = {1: dict(Vmax=1000.0, N=6000, Nrich=(12000, 24000), nlev=500),
          2: dict(Vmax=1500.0, N=8000, Nrich=(16000, 32000), nlev=1000)}
_SETUP = {}


def _setup(scale):
    if scale not in _SETUP:
        S = SCALES[scale]
        L = L_of(S["Vmax"])
        nlev = S["nlev"]
        x = grid(L, S["N"])
        V = wu_sprung_V(x)
        Eg = fd_levels(V, L, nlev)
        Ea = fd_levels(wu_sprung_V(grid(L, S["Nrich"][0])), L, nlev)
        Eb = fd_levels(wu_sprung_V(grid(L, S["Nrich"][1])), L, nlev)
        Erich = richardson(Ea, Eb)
        t = zeros(nlev)
        n = np.arange(1, nlev + 1)
        _SETUP[scale] = dict(L=L, x=x, V=V, E0=Eg, Erich=Erich,
                             target=t + (Eg - Erich), gap=mean_gap(n), nlev=nlev)
    return _SETUP[scale]


def basis(x, L, M, kind="cosine"):
    """Basis functions phi_k(x), k = 0..M-1, as columns: cosine, chebyshev or hat."""
    ax = np.abs(x) / L
    if kind == "cosine":
        return np.cos(np.outer(x, np.arange(M)) * np.pi / L)
    if kind == "chebyshev":
        return np.polynomial.chebyshev.chebvander(2 * ax - 1, M - 1)
    if kind == "hat":
        nodes = np.linspace(0, 1, M)
        w = nodes[1] - nodes[0]
        return np.clip(1 - np.abs(ax[:, None] - nodes[None, :]) / w, 0, None)
    raise ValueError(kind)


def fit_potential(nfit: int, M: int, basis_kind: str = "cosine", ridge: float = 0.0,
                  scale: int = 1, holdout=None, train=None, maxiter: int = 60,
                  lam0: float = 1e-2, accept: float = 0.999, target=None):
    """The paper's Hermitian fit (Sec. II C; App. A step 6), reduced cost.

    Fits delta V = sum_k c_k phi_k to the bias-corrected targets of levels
    `train` (default 1..nfit) and scores levels `holdout` (default nfit+1..nlev).
    `target` (optional) replaces the bias-corrected zeros (for surrogates/probes).
    Returns dict with c, insample_rms, oos_rms, oos_corr, null_oos_rms, E."""
    S = _setup(scale)
    L, x, V, E0, gap, nlev = S["L"], S["x"], S["V"], S["E0"], S["gap"], S["nlev"]
    tgt = S["target"] if target is None else np.asarray(target, float)
    tr = np.arange(1, nfit + 1) if train is None else np.asarray(train)
    ho = np.arange(nfit + 1, nlev + 1) if holdout is None else np.asarray(holdout)
    tri, hoi = tr - 1, ho - 1
    Phi = basis(x, L, M, basis_kind)
    h = 2 * L / (len(x) + 1)
    kmax = int(max(tr.max(), ho.max()))

    def solve(c, vecs=False):
        return fd_levels(V + Phi @ c, L, kmax, return_vectors=vecs)

    def err(E):
        return rms((tgt[tri] - E[tri]) / gap[tri])

    c = np.zeros(M)
    E, psi = solve(c, True)
    cur = err(E)
    lam = lam0
    for _ in range(maxiter):
        A = (psi[:, tri] ** 2).T @ Phi * h            # A_nk = int |psi_n|^2 phi_k dx
        r = tgt[tri] - E[tri]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Utr = U.T @ r
        best = None
        for lt in (0.2 * lam, lam, 5 * lam, 25 * lam):
            filt = s / (s ** 2 + (lt + ridge) * s[0] ** 2)
            cn = c + Vt.T @ (filt * Utr)
            En = solve(cn)
            en = err(En)
            if en < accept * cur and (best is None or en < best[0]):
                best = (en, cn, lt)
        if best is None:
            lam *= 30.0
            if lam > 1e4:
                break
            continue
        cur, c, lacc = best
        lam = max(0.5 * lacc, 1e-7)
        E, psi = solve(c, True)
    pred = (E[hoi] - E0[hoi]) / gap[hoi]
    true = (tgt[hoi] - E0[hoi]) / gap[hoi]
    return dict(c=c, insample_rms=cur, oos_rms=rms((tgt[hoi] - E[hoi]) / gap[hoi]),
                oos_corr=corr(pred, true), null_oos_rms=rms(true), E=E)
