import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
import mpmath
from numpy.polynomial import chebyshev as Cheb

def Nsm(t):
    return t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
def s_n(n):
    return brentq(lambda t: Nsm(t) - (n - 0.5), 7.0, 1e5)
def gapn(n):
    return 2*np.pi/np.log(s_n(n)/(2*np.pi))

# Sample 1: gaps and rms
g = [gapn(1), gapn(2), gapn(3)]
for gv, cl in zip(g, [7.500, 5.279, 4.486]):
    assert abs(gv - cl) < 2e-3, gv
t = np.array([14.134725, 21.022040, 25.010858]); E = np.array([14.30, 20.90, 25.10])
rh = (t - E)/np.array([7.500, 5.279, 4.486])
assert abs(np.sqrt(np.mean(rh**2)) - 0.0217) < 1e-4
assert abs(gapn(500) - 1.3) < 0.05   # "about 1.3"

# Sample 2 / Problem 8: discrete box A exact
L = 44.883; N = 400; h = 2*L/(N+1); x = -L + h*np.arange(1, N+1)
Ev, v = eigh_tridiagonal(2/h**2*np.ones(N), -np.ones(N-1)/h**2, select='i', select_range=(0, 19))
for n in range(1, 21):
    for k in range(0, 31):
        A = np.sum(v[:, n-1]**2*np.cos(k*np.pi*x/L))
        ex = 1.0 if k == 0 else (0.5*(-1)**(n+1) if k == n else 0.0)
        assert abs(abs(A) - abs(ex)) < 1e-10 and (ex == 0 or np.sign(A) == np.sign(ex))
assert abs(1/(2*L/6001) - 67) < 0.5   # pitfall 1/h

# Sample 3: min-norm
Am = np.array([[1, .5, 0, 0], [1, 0, -.5, 0], [1, 0, 0, .5]]); d = np.array([.01, -.02, .03])
c = np.linalg.pinv(Am) @ d
assert np.allclose(c, [0.0061538, 0.0076923, 0.0523077, 0.0476923], atol=1e-6)

# pitfall: spacings, matrix element, second-order size
assert abs(3*np.pi**2/(4*L**2) - 0.00367) < 1e-5 and abs(2*np.pi**2/L**2 - 0.00980) < 1e-5
xx = np.linspace(-L, L, 200001); p = lambda n: np.sin(n*np.pi*(xx+L)/(2*L))/np.sqrt(L)
assert abs(np.trapezoid(p(1)*p(3)*np.cos(2*np.pi*xx/L), xx) + 0.5) < 1e-6
assert abs((0.0523077/2)**2/(2*np.pi**2/L**2) - 0.07) < 0.005

# Wu-Sprung sensitivity text: cutoff for lowest level ~60, E1 - V0 ~ 4.4
V0 = 3.10073*np.pi
assert abs(2*L*np.sqrt(14.134725 - V0)/np.pi - 60) < 0.5
assert abs(14.134725 - V0 - 4.4) < 0.05

# Sample 4: Gauss-Newton
Em = lambda c: 2 - np.sqrt(1 + c**2); Jf = lambda c: -c/np.sqrt(1 + c**2)
cs = [0.5]
for _ in range(3):
    cc = cs[-1]; cs.append(cc + (0.5 - Em(cc))/Jf(cc))
assert np.allclose(cs[1:], [1.354102, 1.126201, 1.118047], atol=2e-6)
cst = np.sqrt(1.25)
assert np.allclose(np.array(cs) - cst, [-0.618, 0.236, 0.0082, 0.000013], atol=[1e-3, 1e-3, 1e-4, 2e-6])
assert abs(2*0.973249*(-0.229753) + 0.447214) < 1e-5
# Newton error constant
Ep = -cst/np.sqrt(1 + cst**2); Epp = -(1 + cst**2)**(-1.5)
assert abs(Ep + 0.7454) < 1e-4 and abs(Epp + 0.2963) < 1e-4
K = abs(Epp/(2*Ep)); assert abs(K - 0.199) < 1e-3
assert Epp/(2*Ep) > 0
assert abs(0.199*0.0082**2 - 1.3e-5) < 0.05e-5
e2 = cs[2] - cst; e3 = cs[3] - cst
assert abs(e3/e2**2 - K) < 0.03

# Sample 5 and 7
Jr = Jf(0.5)*(0.5 - Em(0.5))
for lp, dc, rr in [(0.04, 0.71175, 0.0711), (0.2, 0.42705, 0.1364), (1, 0.14235, 0.3115), (5, 0.03285, 0.3669)]:
    dd = Jr/(0.2 + lp); assert abs(dd - dc) < 1e-5
    assert abs(abs(0.5 - Em(0.5 + dd)) - rr) < 1e-4
assert abs(Jr/5 - 0.03416) < 1e-5
assert abs(abs(0.5 - Em(1.21175)) - 0.071095) < 1e-5 and 0.071095 < 0.999*0.381966

# lm_fit listing (pycode), example call, Problem 17
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
Em2 = lambda c: 2 - np.sqrt(1 + c[0]**2)
res2 = lambda c: np.array([0.5 - Em2(c)])
jac2 = lambda c: np.array([[-c[0]/np.sqrt(1 + c[0]**2)]])
c1, r1 = lm_fit(np.array([0.5]), res2, jac2, maxiter=1)
assert abs(c1[0] - 1.21175) < 1e-5 and abs(r1 - 0.0711) < 1e-4
cf, rf = lm_fit(np.array([0.5]), res2, jac2)
assert abs(cf[0] - cst) < 1e-6 and rf < 1e-6

# Sample 6, Problems 5, 19
f = lambda s, lam, s1: s/(s**2 + lam*s1**2)
assert np.allclose([f(10, .01, 10), f(1, .01, 10), f(.01, .01, 10)], [0.0990, 0.5, 0.0099990], atol=1e-4)
assert np.allclose([f(4, .25, 4), f(2, .25, 4), f(1, .25, 4)], [0.2, 0.25, 0.2])
sg = np.linspace(1e-4, 1, 200001); assert abs(f(sg, .01, 1).max() - 5) < 1e-6

# Sample 8
assert abs(np.cos(3*np.arccos(-0.5)) - 1) < 1e-12 and abs(4*(-0.5)**3 - 3*(-0.5) - 1) < 1e-12
hat = lambda u, j, M: max(0, 1 - (M-1)*abs(u - j/(M-1)))
assert abs(hat(0.6, 2, 5) - 0.6) < 1e-12 and abs(hat(0.6, 3, 5) - 0.4) < 1e-12
assert abs(hat(0.3, 0, 3) - 0.4) < 1e-12 and abs(hat(0.3, 1, 3) - 0.6) < 1e-12   # Problem 13

# Chebyshev kink: T_k'(-1) = (-1)^(k+1) k^2, slope of phi_k at 0+ = (2/L) T_k'(-1)
for k in range(1, 8):
    dcoef = Cheb.chebder([0]*k + [1])
    assert abs(Cheb.chebval(-1.0, dcoef) - (-1)**(k+1)*k**2) < 1e-9
    eps = 1e-6
    sl = (Cheb.chebval(2*eps/L - 1, [0]*k + [1]) - Cheb.chebval(-1.0, [0]*k + [1]))/eps
    assert abs(sl - 2*(-1)**(k+1)*k**2/L) < 1e-3*abs(2*k**2/L) + 1e-6

# Stationary-phase illustration, a = 100, X = 3
a = 100.0; X = 3.0
xs = np.linspace(-X, X, 2000001)
lin = np.trapezoid(np.cos(a*xs), xs)
assert abs(lin - 2*np.sin(a*X)/a) < 1e-6 and abs(lin) <= 0.02 + 1e-9
quad = np.trapezoid(np.cos(a*xs**2), xs)
assert abs(quad - 0.13) < 0.005
assert abs(np.sqrt(np.pi/(2*a)) - 0.125) < 0.001

# Sample 9, coverage text and figure
assert abs(2*np.sqrt(811.1843588465)*L/np.pi - 813.8) < 0.2
Emax = (400*np.pi/(2*L))**2; assert abs(Emax - 196.0) < 0.1
assert abs(Nsm(196) - 77.0) < 0.2
assert 600*np.pi/L < 2*np.sqrt(811.1843588465) < 900*np.pi/L   # only M=900 clears 500 levels
assert abs(V0 - 9.7412) < 1e-3
assert abs(V0/(2*811.1843588465) - 0.006) < 2e-4               # 0.6%
assert abs(np.sqrt(1 - V0/14.134725) - 0.56) < 0.005

# Coverage reconciliation: t_400 ~ 679.7, law asks M >~ 745
t400 = float(mpmath.zetazero(400).imag)
assert abs(t400 - 679.7) < 0.05
assert abs(2*np.sqrt(t400)*L/np.pi - 745) < 1

# Problem 18: cutoff for n=50 lies between 300 and 500
t50 = float(mpmath.zetazero(50).imag)
cut50 = 2*L*np.sqrt(t50 - V0)/np.pi
assert 300 < cut50 < 499

# Problem 11 (scale 2, 800 training levels)
s800 = s_n(800); assert abs(s800 - 1184) < 1
assert abs(2*np.sqrt(s800) - 68.8) < 0.05 and abs(59.957/np.pi - 19.085) < 1e-3
assert abs(2*np.sqrt(s800)*59.957/np.pi - 1313) < 2
E900 = (900*np.pi/(2*59.957))**2
assert abs(900*np.pi/(2*59.957) - 23.58) < 0.01 and abs(E900 - 556) < 1
assert abs(Nsm(E900) - 309) < 1

# Problems 2, 7
assert abs(np.sqrt(0.03) - 0.1732) < 1e-4
c = 1.0
for cl in [1.5, 1.41667]:
    c = c + (2 - c**2)/(2*c); assert abs(c - cl) < 1e-5

# Problem 24: rank of first-order box model, L=10, N=300
Lb = 10.0; Nb = 300; hb = 2*Lb/(Nb+1); xb = -Lb + hb*np.arange(1, Nb+1)
wb, vb = eigh_tridiagonal(2/hb**2*np.ones(Nb), -np.ones(Nb-1)/hb**2, select='i', select_range=(0, 30))
Mm = 10
Phb = np.cos(np.outer(xb, np.arange(Mm))*np.pi/Lb)
Ab = (vb**2).T @ Phb
assert np.linalg.matrix_rank(Ab[:Mm], tol=1e-8) == Mm
assert np.linalg.matrix_rank(Ab[:Mm+1], tol=1e-8) == Mm

# Toy fit figure: collapse at M = nfit = 30
nfit = 30
E0 = wb
gapb = np.gradient(E0)[:nfit]
rng = np.random.default_rng(0); dd = rng.standard_normal(nfit); dd *= 0.2/np.sqrt(np.mean(dd**2))
tgt = E0[:nfit] + dd*gapb
def toy_rms(M):
    Ph = np.cos(np.outer(xb, np.arange(M))*np.pi/Lb)
    def resid(c):
        w = eigh_tridiagonal(2/hb**2 + Ph @ c, -np.ones(Nb-1)/hb**2, select='i', select_range=(0, nfit-1), eigvals_only=True)
        return (tgt - w)/gapb
    def jac(c):
        w, vv = eigh_tridiagonal(2/hb**2 + Ph @ c, -np.ones(Nb-1)/hb**2, select='i', select_range=(0, nfit-1))
        return ((vv**2).T @ Ph)/gapb[:, None]
    return lm_fit(np.zeros(M), resid, jac)[1]
r10, r20, r30 = toy_rms(10), toy_rms(20), toy_rms(30)
assert r10 > r20 > r30 and r30 < 1e-6 and r20 > 1e-3
print("all checks passed")
