import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

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

# Sample 4: Gauss-Newton
Em = lambda c: 2 - np.sqrt(1 + c**2); Jf = lambda c: -c/np.sqrt(1 + c**2)
cs = [0.5]
for _ in range(3):
    cc = cs[-1]; cs.append(cc + (0.5 - Em(cc))/Jf(cc))
assert np.allclose(cs[1:], [1.354102, 1.126201, 1.118047], atol=2e-6)
cst = np.sqrt(1.25)
assert np.allclose(np.array(cs) - cst, [-0.618, 0.236, 0.0082, 0.000013], atol=[1e-3, 1e-3, 1e-4, 2e-6])
assert abs(2*0.973249*(-0.229753) + 0.447214) < 1e-5

# Sample 5 and 7
Jr = Jf(0.5)*(0.5 - Em(0.5))
for lp, dc, rr in [(0.04, 0.71175, 0.0711), (0.2, 0.42705, 0.1364), (1, 0.14235, 0.3115), (5, 0.03285, 0.3669)]:
    dd = Jr/(0.2 + lp); assert abs(dd - dc) < 1e-5
    assert abs(abs(0.5 - Em(0.5 + dd)) - rr) < 1e-4
assert abs(Jr/5 - 0.03416) < 1e-5
assert abs(abs(0.5 - Em(1.21175)) - 0.071095) < 1e-5 and 0.071095 < 0.999*0.381966

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

# Sample 9, Problem 11
assert abs(2*np.sqrt(811.1843588465)*L/np.pi - 813.8) < 0.2
Emax = (400*np.pi/(2*L))**2; assert abs(Emax - 196.0) < 0.1
assert abs(Nsm(196) - 77.0) < 0.2
assert abs(2*np.sqrt(1419.42248094599)*59.957/np.pi - 1438) < 1
assert abs(3.10073*np.pi - 9.7412) < 1e-3

# Problems 2, 7
assert abs(np.sqrt(0.03) - 0.1732) < 1e-4
c = 1.0
for cl in [1.5, 1.41667]:
    c = c + (2 - c**2)/(2*c); assert abs(c - cl) < 1e-5
print("all checks passed")
