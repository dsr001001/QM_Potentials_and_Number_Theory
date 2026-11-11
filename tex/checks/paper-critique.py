import math
import numpy as np
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

t = qmnt.zeros(1000)

# ---- Sample "Relative skill of the operator and of the oracle"
assert abs((1 - 0.369 / 0.263) - (-0.403)) < 1e-3
assert abs(0.369 / 0.263 - 1.403) < 1e-3
assert abs((1 - 0.074 / 0.263) - 0.719) < 1e-3
assert abs(0.074 / 0.263 - 0.281) < 1e-3

# ---- Sample "What a correlation is worth"
assert abs(math.sqrt(1 - 0.96**2) - 0.280) < 1e-3
assert abs(0.263 * math.sqrt(1 - 0.96**2) - 0.0736) < 1e-4
assert abs(math.sqrt(1 - 0.07**2) - 0.9975) < 1e-4
assert abs(0.263 * math.sqrt(1 - 0.07**2) - 0.2624) < 1e-4
rng = np.random.default_rng(3)
f = rng.normal(size=200); f -= f.mean()
g = 0.7 * f + rng.normal(size=200); g -= g.mean()
a = np.sum(f * g) / np.sum(g * g)
Qmin = np.mean((f - a * g) ** 2)
r = np.corrcoef(f, g)[0, 1]
assert abs(Qmin - np.mean(f**2) * (1 - r**2)) < 1e-12

# ---- Sample "Coverage threshold at the two scales"
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
# inverted coverage law at scale 2, M = 900
assert abs(2 * 59.957 - 119.914) < 1e-9
k900 = 900 * np.pi / (2 * L2)
assert abs(k900 - 23.58) < 0.01
E900 = k900 ** 2
assert abs(E900 - 556) < 1.0
assert abs(float(np.asarray(qmnt.Nsm(556.0)).ravel()[0]) + 0.5 - 310) < 2

# ---- Sample "Why item (ii) matters" (improvement over null, positive = better)
assert abs((0.2438 - 0.2321) / 0.2438 - 0.0480) < 1e-4
assert abs((0.2438 - 0.2523) / 0.2438 - (-0.0349)) < 1e-4
assert abs((0.2438 - 0.2358) / 0.2438 - 0.0328) < 1e-4
assert abs((0.2438 - 0.2477) / 0.2438 - (-0.0160)) < 1e-4

# ---- Sample "One marginal row among eight"
assert abs(0.975**8 - 0.8167) < 1e-4
assert abs(1 - 0.975**8 - 0.183) < 1e-3
assert abs(8 * 0.025 - 0.2) < 1e-12

# ---- Sample "Worst-case dephasing"
assert abs(t[0] - 14.1347251417) < 1e-8
assert abs(t[499] - 811.1843588465) < 1e-8
assert abs(t[999] - 1419.42248094599) < 1e-8
T500 = 811.1843588465 - 14.1347251417
assert abs(T500 - 797.0496) < 1e-4
dw = np.pi / T500
assert abs(dw - 0.003942) < 1e-6
dist = 1419.42248094599 - 811.1843588465
assert abs(dist - 608.2381) < 1e-4
dphi1 = dw * dist
assert abs(dphi1 - 2.397) < 1e-3
assert abs(dphi1 / (2 * np.pi) - 0.38) < 5e-3
assert abs(3 * dphi1 - 7.19) < 5e-3
assert abs(2 * np.pi / np.log(2) - 9.065) < 1e-3

# ---- Sample "What none out of n can support"
f29 = 1 - 0.05 ** (1 / 29)
f57 = 1 - 0.05 ** (1 / 57)
assert abs(math.log(0.05) - (-2.996)) < 1e-3
assert abs(math.log(0.05) / 29 - (-0.10330)) < 1e-5
assert abs(math.exp(math.log(0.05) / 29) - 0.90186) < 1e-5
assert abs(f29 - 0.0981) < 1e-4
assert abs(3 / 29 - 0.103) < 1e-3
assert abs(math.log(0.05) / 57 - (-0.05256)) < 1e-5
assert abs(math.exp(math.log(0.05) / 57) - 0.94880) < 1e-5
assert abs(f57 - 0.0512) < 1e-4
assert abs(3 / 57 - 0.053) < 1e-3

# ---- Sample "Turning points of the interpolation gap"
s201, s300 = sz([201, 300])
assert abs(s201 - 398) < 1.5 and abs(s300 - 543) < 1.5
x201, x300 = wsx([s201, s300])
assert abs(x201 - 22.5) < 0.2
assert abs(x300 - 28.5) < 0.2

# ---- W4 three-level example
E = np.array([1.0, 2.0, 4.0])
A = np.array([[0.0, 0.05, 0.1], [0.05, 0.0, 0.15], [0.1, 0.15, 0.0]])
dE = np.array([-sum(A[i, j]**2 / (E[i] - E[j]) for j in range(3) if j != i) for i in range(3)])
assert np.allclose(dE, [0.005833, 0.00875, -0.014583], atol=1e-6)
assert abs(dE.sum()) < 1e-15
lam = np.linalg.eigvals(np.diag(E) + 1j * A)
lam = lam[np.argsort(lam.real)]
assert np.max(np.abs(lam.real - (E + dE))) < 1e-3
assert abs(lam.real.sum() - 7.0) < 1e-12
A2 = np.array([[0.3, 0.2, -0.1], [0.2, -0.4, 0.25], [-0.1, 0.25, 0.1]])
lam2 = np.linalg.eigvals(np.diag(E) + 1j * A2)
assert abs(lam2.real.sum() - E.sum()) < 1e-12

# ---- Sample "How often does a random frequency land near some log q?"
t400 = qmnt.zeros(400)
T400 = t400[-1] - t400[0]
assert abs(T400 - 666) < 3
c = 2 * np.pi / 666
assert abs(c - 0.009434) < 1e-6
m2 = c * np.exp(2) / 2; m5 = c * np.exp(5) / 5; m8 = c * np.exp(8) / 8
assert abs(np.exp(2) / 2 - 3.695) < 1e-3 and abs(m2 - 0.0349) < 1e-4 and abs(1 - np.exp(-m2) - 0.034) < 1e-3
assert abs(np.exp(5) / 5 - 29.68) < 1e-2 and abs(m5 - 0.280) < 1e-3 and abs(1 - np.exp(-m5) - 0.244) < 1e-3
assert abs(np.exp(8) / 8 - 372.6) < 0.1 and abs(m8 - 3.52) < 5e-3 and abs(1 - np.exp(-m8) - 0.970) < 1e-3
assert abs(np.log(3000) - 8.006) < 1e-3

# ---- Sample "Reproducing the cheap anchors"
nh = np.arange(401, 501)
null_401 = qmnt.rms(qmnt.fluct(t[nh - 1], nh))
assert abs(null_401 - 0.2631) < 6e-4

# ---- Sample "Reproducing the six-prime oracle"
r6, c6 = qmnt.score(qmnt.oracle_zeros(nh, qmnt.first_primes(6)), nh)
assert abs(r6 - 0.074) < 1.5e-3
assert abs(c6 - 0.96) < 6e-3

# ---- Leakage test listing
n5 = np.arange(1, 501)
tt5 = qmnt.zeros(500).copy()
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

# ---- Problem 2
assert abs((1 - 0.313 / 0.263) - (-0.190)) < 1e-3
assert abs((1 - 0.258 / 0.263) - 0.019) < 1e-3
assert abs((1 - 0.028 / 0.263) - 0.894) < 1e-3

# ---- Problem 3
assert abs(2.64 / 0.255 - 10.35) < 1e-2
assert abs(2.6434 / 0.255 - 10.37) < 1e-2

# ---- Problem 7
assert abs((0.2438 - 0.2391) / 0.2438 - 0.0193) < 1e-4
assert abs((0.2438 - 0.2423) / 0.2438 - 0.0062) < 1e-4

# ---- Problem 9
n2 = np.arange(501, 1001)
null_501 = qmnt.rms(qmnt.fluct(t[n2 - 1], n2))
assert abs(null_401 - 0.263) < 1e-3
assert abs(null_501 - 0.270) < 1.5e-3

# ---- Problem 11
s401, s500 = sz([401, 500])
assert abs(s401 - 681) < 1.5 and abs(s500 - 811) < 1.5
x401, x500 = wsx([s401, s500])
assert abs(x401 - 33.9) < 0.3
assert abs(x500 - 38.5) < 0.3
assert abs(33.9 / 44.883 - 0.755) < 1e-3
assert abs(38.5 / 44.883 - 0.858) < 1e-3

# ---- Problem 13
m3 = c * np.exp(3) / 3; m6 = c * np.exp(6) / 6
assert abs(np.exp(3) / 3 - 6.695) < 1e-3 and abs(m3 - 0.0632) < 1e-4
assert abs(1 - np.exp(-m3) - 0.061) < 1e-3
assert abs(np.exp(6) / 6 - 67.24) < 1e-2 and abs(m6 - 0.634) < 1e-3
assert abs(1 - np.exp(-m6) - 0.470) < 1e-3

# ---- Problem 17
assert abs((0.137 - 0.134) / 0.010 - 0.3) < 1e-9
assert abs((0.141 - 0.137) / 0.006 - 0.67) < 5e-3
assert abs((0.138 - 0.137) / 0.005 - 0.2) < 1e-9

# ---- Problem 19
T800 = t[799] - t[0]
assert abs(T800 - 1170) < 3
assert abs(np.log(131) - 4.875) < 1e-3
assert abs(131 / np.log(131) - 26.87) < 1e-2
c800 = 2 * np.pi / 1170
assert abs(c800 - 0.005370) < 1e-6
m131 = c800 * 131 / np.log(131)
assert abs(m131 - 0.144) < 1e-3
assert abs(1 - np.exp(-m131) - 0.134) < 1e-3

# ---- Problem 23
assert abs((1 - 0.05 ** (1 / 29)) / 0.8 - 0.123) < 1e-3

# ---- Problem 25 and W8 factor
assert abs(0.265 / 0.198 - 1.34) < 1e-2
assert abs(0.264 / 0.198 - 1.33) < 1e-2

print("all checks passed")
