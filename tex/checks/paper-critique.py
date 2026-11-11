import math
import numpy as np
import scipy.special
import qmnt

def sz(n):
    n = np.atleast_1d(np.asarray(n))
    try:
        out = np.asarray(qmnt.smooth_zeros(n), dtype=float)
        if out.shape != n.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.smooth_zeros(int(k))) for k in n])

def wsx(V):
    V = np.atleast_1d(np.asarray(V, dtype=float))
    try:
        out = np.asarray(qmnt.wu_sprung_x(V), dtype=float)
        if out.shape != V.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.wu_sprung_x(v)) for v in V])

t2 = np.asarray(qmnt.zeros(2000), dtype=float)
t = t2[:1000]

# ---- Sample relative skill
assert abs((1 - 0.369 / 0.263) - (-0.403)) < 1e-3
assert abs(0.369 / 0.263 - 1.403) < 1e-3
assert abs((1 - 0.074 / 0.263) - 0.719) < 1e-3
assert abs(0.074 / 0.263 - 0.281) < 1e-3

# ---- Sample what a correlation is worth
assert abs(math.sqrt(1 - 0.96**2) - 0.280) < 1e-3
assert abs(0.263 * math.sqrt(1 - 0.96**2) - 0.0736) < 1e-4
assert abs(math.sqrt(1 - 0.07**2) - 0.9975) < 1e-4
assert abs(0.263 * math.sqrt(1 - 0.07**2) - 0.2624) < 1e-4
assert abs(0.263 * math.sqrt(1 - 0.955**2) - 0.078) < 5e-4
assert abs(0.263 * math.sqrt(1 - 0.965**2) - 0.069) < 5e-4
rng = np.random.default_rng(3)
f = rng.normal(size=200); f -= f.mean()
g = 0.7 * f + rng.normal(size=200); g -= g.mean()
a = np.sum(f * g) / np.sum(g * g)
Qmin = np.mean((f - a * g) ** 2)
r = np.corrcoef(f, g)[0, 1]
assert abs(Qmin - np.mean(f**2) * (1 - r**2)) < 1e-12
nh = np.arange(401, 501)
fl = np.asarray(qmnt.fluct(t[nh - 1], nh), dtype=float)
assert abs(np.mean(fl)) < 0.2 * qmnt.rms(fl)   # mean less than a fifth of rms

# ---- Coverage sample
L1 = qmnt.L_of(1000); L2 = qmnt.L_of(1500)
assert abs(L1 - 44.883) < 1e-3 and abs(L2 - 59.957) < 1e-3
assert abs(math.sqrt(811.18) - 28.481) < 1e-3
assert abs(2 * math.sqrt(811.18) - 56.963) < 2e-3
Mth1 = 2 * np.sqrt(t[499]) * L1 / np.pi
assert abs(Mth1 - 813.8) < 0.2
s800 = sz(800)[0]
assert abs(s800 - 1184) < 1.0
Mth2 = 2 * np.sqrt(s800) * L2 / np.pi
assert abs(Mth2 - 1313) < 3
assert abs(2 * 59.957 - 119.914) < 1e-9
k900 = 900 * np.pi / (2 * L2)
assert abs(k900 - 23.58) < 0.01
assert abs(k900**2 - 556) < 1.0
assert abs(float(np.asarray(qmnt.Nsm(556.0)).ravel()[0]) + 0.5 - 310) < 2
# W3: scale-1 training threshold and ratios
s400 = sz(400)[0]
assert abs(s400 - 680) < 1
Mth1tr = 2 * np.sqrt(s400) * L1 / np.pi
assert abs(Mth1tr - 745) < 1.5
assert abs(2 * np.sqrt(680) * 44.883 / np.pi - 745) < 1.5
for M, thr, claim in ((200, Mth1tr, 0.27), (600, Mth1tr, 0.81), (300, Mth2, 0.23), (900, Mth2, 0.69)):
    assert abs(M / thr - claim) < 6e-3
assert 900 > Mth1

# ---- Checkpoint hypothetical paper
assert abs((1 - 0.262 / 0.263) - 0.004) < 5e-4
assert abs(1 / math.sqrt(100) - 0.1) < 1e-12

# ---- Item (ii) sample
assert abs((0.2438 - 0.2321) / 0.2438 - 0.0480) < 1e-4
assert abs((0.2438 - 0.2523) / 0.2438 - (-0.0349)) < 1e-4
assert abs((0.2438 - 0.2358) / 0.2438 - 0.0328) < 1e-4
assert abs((0.2438 - 0.2477) / 0.2438 - (-0.0160)) < 1e-4

# ---- Marginal row sample and two-sided remark
assert abs(0.975**8 - 0.8167) < 1e-4
assert abs(1 - 0.975**8 - 0.183) < 1e-3
assert abs(8 * 0.025 - 0.2) < 1e-12
assert abs(1 - 0.95**8 - 0.34) < 5e-3

# ---- Anchors
assert abs(t[0] - 14.1347251417) < 1e-8
assert abs(t[499] - 811.1843588465) < 1e-8
assert abs(t[999] - 1419.42248094599) < 1e-8
assert abs(2 * np.pi / np.log(2) - 9.065) < 1e-3

# ---- Dephasing sample (window zeros 1--400)
T400 = t2[399] - t2[0]
assert abs(T400 - 666) < 3
dw4 = np.pi / T400
assert abs(dw4 - 0.00472) < 2e-5
d1001 = t2[1000] - t2[399]
d2000 = t2[1999] - t2[399]
assert abs(d1001 - 741) < 2
assert abs(d2000 - 1836) < 2
assert abs(dw4 * d1001 - 3.5) < 0.05
assert abs(dw4 * d2000 - 8.7) < 0.07
assert abs(0.00472 * 741 - 3.5) < 0.01
assert abs(0.00472 * 1836 - 8.7) < 0.05
assert abs(dw4 * T400 - np.pi) < 1e-12
assert abs(0.1 * dw4 * d2000 - 0.87) < 0.01

# ---- Rule of three sample
f29 = 1 - 0.05 ** (1 / 29)
f57 = 1 - 0.05 ** (1 / 57)
assert abs(math.log(0.05) - (-2.996)) < 1e-3
assert abs(math.log(0.05) / 29 - (-0.10330)) < 1e-5
assert abs(math.exp(math.log(0.05) / 29) - 0.90186) < 1e-5
assert abs(f29 - 0.0981) < 1e-4 and f29 < 0.1
assert abs(3 / 29 - 0.103) < 1e-3
assert abs(math.log(0.05) / 57 - (-0.05256)) < 1e-5
assert abs(math.exp(math.log(0.05) / 57) - 0.94880) < 1e-5
assert abs(f57 - 0.0512) < 1e-4 and abs(f57 - 0.05) < 2e-3
assert abs(3 / 57 - 0.053) < 1e-3

# ---- Turning points sample
s201, s300 = sz([201, 300])
assert abs(s201 - 398) < 1.5 and abs(s300 - 543) < 1.5
x201, x300 = wsx([s201, s300])
assert abs(x201 - 22.5) < 0.2
assert abs(x300 - 28.5) < 0.2

# ---- W4 three-level example and epsilon expansion
E = np.array([1.0, 2.0, 4.0])
A = np.array([[0.0, 0.05, 0.1], [0.05, 0.0, 0.15], [0.1, 0.15, 0.0]])
dE = np.array([-sum(A[i, j]**2 / (E[i] - E[j]) for j in range(3) if j != i) for i in range(3)])
assert np.allclose(dE, [0.005833, 0.00875, -0.014583], atol=1e-6)
assert abs(dE.sum()) < 1e-15
lam = np.linalg.eigvals(np.diag(E) + 1j * A)
lam = lam[np.argsort(lam.real)]
assert np.max(np.abs(lam.real - (E + dE))) < 1e-3
assert abs(lam.real.sum() - 7.0) < 1e-12
eps = 1e-3
lame = np.linalg.eigvals(np.diag(E) + 1j * eps * A)
lame = lame[np.argsort(lame.real)]
assert np.allclose((lame.real - E) / eps**2, dE, atol=1e-5)

# ---- Problem 15 example
A2 = np.array([[0.3, 0.2, -0.1], [0.2, -0.4, 0.25], [-0.1, 0.25, 0.1]])
dE2 = np.array([-sum(A2[i, j]**2 / (E[i] - E[j]) for j in range(3) if j != i) for i in range(3)])
assert np.allclose(dE2, [0.04333, -0.00875, -0.03458], atol=1e-5)
assert abs(dE2.sum()) < 1e-14
lam2 = np.linalg.eigvals(np.diag(E) + 1j * A2)
assert abs(lam2.real.sum() - E.sum()) < 1e-12
assert abs(np.trace(A2)) < 1e-12

# ---- Chance-match sample
t400 = t2[:400]
assert abs((t400[-1] - t400[0]) - 666) < 3
c = 2 * np.pi / 666
assert abs(c - 0.009434) < 1e-6
m2 = c * np.exp(2) / 2; m5 = c * np.exp(5) / 5; m8 = c * np.exp(8) / 8
assert abs(np.exp(2) / 2 - 3.695) < 1e-3 and abs(m2 - 0.0349) < 1e-4 and abs(1 - np.exp(-m2) - 0.034) < 1e-3
assert abs(np.exp(5) / 5 - 29.68) < 1e-2 and abs(m5 - 0.280) < 1e-3 and abs(1 - np.exp(-m5) - 0.244) < 1e-3
assert abs(np.exp(8) / 8 - 372.6) < 0.1 and abs(m8 - 3.52) < 5e-3 and abs(1 - np.exp(-m8) - 0.970) < 1e-3
assert abs(np.log(3000) - 8.006) < 1e-3

# ---- Null and six-prime oracle
null_401 = qmnt.rms(qmnt.fluct(t[nh - 1], nh))
assert abs(null_401 - 0.2631) < 6e-4
r6, c6 = qmnt.score(qmnt.oracle_zeros(nh, qmnt.first_primes(6)), nh)
assert abs(r6 - 0.074) < 1.5e-3
assert abs(c6 - 0.96) < 6e-3

# ---- Height figure caption: relative skill at first and last holdout
assert abs((1 - 0.074 / 0.263) - 0.72) < 5e-3
assert abs((1 - 0.119 / 0.296) - 0.60) < 5e-3
assert abs((1 - 0.043 / 0.263) - 0.84) < 5e-3
assert abs((1 - 0.067 / 0.296) - 0.77) < 5e-3

# ---- Leakage test listing
n5 = np.arange(1, 501)
tt5 = t2[:500].copy()
train = n5 <= 400
P3 = qmnt.first_primes(3)
omegas = np.array([m * np.log(p) for p in P3 for m in (1, 2, 3)])
def fitB(heights):
    y = n5 - 0.5 - qmnt.Nsm(heights)
    tt, yy = heights[train], y[train]
    X = np.hstack([np.sin(np.outer(tt, omegas)), np.cos(np.outer(tt, omegas))])
    return np.linalg.lstsq(X, yy, rcond=None)[0]
cA = fitB(tt5)
rngL = np.random.default_rng(0)
tn = tt5.copy()
tn[~train] = rngL.uniform(0, 5000, size=100)
cB = fitB(tn)
assert np.array_equal(cA, cB)

# ---- Project 5 Hermite polynomials
assert abs(scipy.special.eval_hermite(2, 0.7) - (4 * 0.49 - 2)) < 1e-12
assert abs(scipy.special.eval_hermite(3, 0.7) - (2 * 0.7 * scipy.special.eval_hermite(2, 0.7) - 4 * scipy.special.eval_hermite(1, 0.7))) < 1e-12

# ---- Problem 2
assert abs((1 - 0.313 / 0.263) - (-0.190)) < 1e-3
assert abs((1 - 0.258 / 0.263) - 0.019) < 1e-3
assert abs((1 - 0.028 / 0.263) - 0.894) < 1e-3

# ---- Problem 3
assert abs(2.64 / 0.255 - 10.35) < 1e-2
assert abs(2.6434 / 0.255 - 10.37) < 1e-2

# ---- Problem 6 (factor 1.3)
assert abs(0.265 / 0.198 - 1.34) < 1e-2
assert abs(0.264 / 0.198 - 1.33) < 1e-2

# ---- Problem 7
assert abs((0.1958 - 0.1708) / 0.1708 - 0.146) < 1e-3

# ---- Problem 8
assert abs((0.2438 - 0.2391) / 0.2438 - 0.0193) < 1e-4
assert abs((0.2438 - 0.2423) / 0.2438 - 0.0062) < 1e-4

# ---- Problem 9
assert abs(0.975**12 - 0.738) < 1e-3
assert abs(1 - 0.975**12 - 0.262) < 1e-3

# ---- Problem 10
n2 = np.arange(501, 1001)
null_501 = qmnt.rms(qmnt.fluct(t[n2 - 1], n2))
assert abs(null_401 - 0.263) < 1e-3
assert abs(null_501 - 0.270) < 1.5e-3

# ---- Problem 11
r1, c1 = qmnt.score(qmnt.oracle_zeros(nh, qmnt.first_primes(1)), nh)
r3, c3 = qmnt.score(qmnt.oracle_zeros(nh, qmnt.first_primes(3)), nh)
assert abs(r1 - 0.198) < 1.5e-3 and abs(c1 - 0.66) < 6e-3
assert abs(r3 - 0.116) < 1.5e-3 and abs(c3 - 0.90) < 6e-3
assert abs((1 - 0.198 / 0.263) - 0.247) < 1e-3
assert abs((1 - 0.116 / 0.263) - 0.559) < 1e-3

# ---- Problem 12
s401, s500 = sz([401, 500])
assert abs(s401 - 681) < 1.5 and abs(s500 - 811) < 1.5
x401, x500 = wsx([s401, s500])
assert abs(x401 - 33.9) < 0.3
assert abs(x500 - 38.5) < 0.3
assert abs(33.9 / 44.883 - 0.755) < 1e-3
assert abs(38.5 / 44.883 - 0.858) < 1e-3

# ---- Problem 13
assert abs(666 / 2 - 333) < 1e-12
assert abs((np.pi / 666) * 333 - np.pi / 2) < 1e-12

# ---- Problem 14
m3 = c * np.exp(3) / 3; m6 = c * np.exp(6) / 6
assert abs(np.exp(3) / 3 - 6.695) < 1e-3 and abs(m3 - 0.0632) < 1e-4
assert abs(1 - np.exp(-m3) - 0.061) < 1e-3
assert abs(np.exp(6) / 6 - 67.24) < 1e-2 and abs(m6 - 0.634) < 1e-3
assert abs(1 - np.exp(-m6) - 0.470) < 1e-3

# ---- Problem 18
assert abs((0.137 - 0.134) / 0.010 - 0.3) < 1e-9
assert abs((0.141 - 0.137) / 0.006 - 0.67) < 5e-3
assert abs((0.138 - 0.137) / 0.005 - 0.2) < 1e-9

# ---- Problem 21
T800 = t[799] - t[0]
assert abs(T800 - 1170) < 3
assert abs(np.log(131) - 4.875) < 1e-3
assert abs(131 / np.log(131) - 26.87) < 1e-2
c800 = 2 * np.pi / 1170
assert abs(c800 - 0.005370) < 1e-6
m131 = c800 * 131 / np.log(131)
assert abs(m131 - 0.144) < 1e-3
assert abs(1 - np.exp(-m131) - 0.134) < 1e-3

# ---- Problem 25
assert abs((1 - 0.05 ** (1 / 29)) / 0.8 - 0.123) < 1e-3

print("all checks passed")
