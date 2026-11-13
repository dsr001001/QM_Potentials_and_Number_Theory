import math
from math import log, sqrt, pi, comb
from functools import reduce
import numpy as np
import mpmath
from scipy.special import expi
import qmnt


def sieve(x):
    s = np.ones(x + 1, dtype=bool)
    s[:2] = False
    p = 2
    while p * p <= x:
        if s[p]:
            s[p * p::p] = False
        p += 1
    return np.nonzero(s)[0]


def isprime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def factor(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def Lam(k):
    if k < 2:
        return 0.0
    f = factor(k)
    return log(list(f)[0]) if len(f) == 1 else 0.0


P = sieve(10**6)
assert np.array_equal(P, np.asarray(qmnt.primes_upto(10**6)))
pi_of = lambda x: int(np.searchsorted(P, x, side="right"))

# ---- Section 1, Sample: 221 and 1009; checkpoint 10403
assert abs(sqrt(221) - 14.87) < 0.01 and 13 * 17 == 221
assert abs(sqrt(1009) - 31.76) < 0.01 and isprime(1009)
assert [1009 % p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]] == [1, 1, 4, 1, 8, 8, 6, 2, 20, 23, 17]
assert 221 == 7 * 31 + 4 and 221 == 11 * 20 + 1 and 1009 == 31 * 32 + 17
assert abs(sqrt(10403) - 101.99) < 0.01 and 101 * 103 == 10403

# ---- Sample: 5040
assert factor(5040) == {2: 4, 3: 2, 5: 1, 7: 1}
assert abs(log(5040) - 8.5252) < 1e-4
assert abs(4 * log(2) - 2.7726) < 1e-4 and abs(2 * log(3) - 2.1972) < 1e-4
assert abs(log(5) - 1.6094) < 1e-4 and abs(log(7) - 1.9459) < 1e-4
assert abs((2.7726 + 2.1972 + 1.6094 + 1.9459) - 8.5251) < 1e-9

# ---- Euclid numbers (Sample and Problem 7)
E = []
prod = 1
for p in [2, 3, 5, 7, 11, 13]:
    prod *= p
    E.append(prod + 1)
assert E == [3, 7, 31, 211, 2311, 30031]
assert all(isprime(e) for e in E[:5]) and not isprime(30031)
assert 59 * 509 == 30031 and isprime(59) and isprime(509)
assert sqrt(211) < 15 and sqrt(2311) < 49
assert all(not isprime(math.factorial(5) + j) for j in range(2, 6))

# ---- Sieve sample: 15 primes up to 50, 25 up to 100; checkpoint 200 -> 13
assert pi_of(50) == 15 and pi_of(100) == 25 and P[24] == 97
assert 13 ** 2 <= 200 < 17 ** 2

# ---- pi(x) facts
assert pi_of(10) == 4 and pi_of(30) == 10 and pi_of(20) == 8
assert [int(p) for p in P[:6]] == [2, 3, 5, 7, 11, 13]

# ---- Table of pi(x), x/log x, Li(x)
tab = {10: (4, 4.343, 5.120, 0.921, -1.12, 2.500),
       10**2: (25, 21.715, 29.081, 1.151, -4.08, 4.000),
       10**3: (168, 144.765, 176.564, 1.161, -8.56, 5.952),
       10**4: (1229, 1085.736, 1245.092, 1.132, -16.09, 8.137),
       10**5: (9592, 8685.890, 9628.764, 1.104, -36.76, 10.425),
       10**6: (78498, 72382.414, 78626.504, 1.084, -128.50, 12.739)}
Lis = {}
for x, (pc, xl, li, r, d, sp) in tab.items():
    Li = float(mpmath.li(x, offset=True))
    Lis[x] = Li
    assert pi_of(x) == pc
    assert abs(x / log(x) - xl) < 1e-3
    assert abs(Li - li) < 1e-3
    assert abs(pc / (x / log(x)) - r) < 1e-3
    assert abs(pc - Li - d) < 0.01
    assert abs(x / pc - sp) < 1e-3
    assert pc < Li
incs = [tab[10**(e + 1)][5] - tab[10**e][5] for e in range(1, 6)]
for a, b in zip(incs, [1.50, 1.95, 2.18, 2.29, 2.31]):
    assert abs(a - b) < 0.006
assert abs(log(10) - 2.3026) < 1e-4
assert abs(float(mpmath.li(2)) - 1.045) < 1e-3

# ---- Sample: relative errors; interval [1e4,1e5]
assert abs((144.765 - 168) / 168 - (-0.138)) < 1e-3
assert abs((176.564 - 168) / 168 - 0.051) < 1e-3
assert abs((72382.414 - 78498) / 78498 - (-0.078)) < 1e-3
assert abs((78626.504 - 78498) / 78498 - 0.0016) < 1e-4
assert abs(Lis[10**5] - Lis[10**4] - 8383.67) < 0.01
assert pi_of(10**5) - pi_of(10**4) == 8363
# small-x errors (revised sentence after the sample)
assert abs(10 / log(10) - 4 - 0.34) < 0.005 and abs(100 / log(100) - 25 - (-3.29)) < 0.005
assert abs(Lis[10] - 4 - 1.12) < 0.005 and abs(Lis[100] - 25 - 4.08) < 0.005
# 1/log x at 1e6 and mean spacing
assert abs(1 / log(1e6) - 0.072) < 1e-3
assert abs(log(1e6) - 13.82) < 0.01 and abs(1e6 / 78498 - 12.74) < 0.01
assert abs(78498 - 72382.414 - 6115.6) < 0.1
assert abs(2 / log(2) - 2.885) < 1e-3

# ---- pi(x) < li(x) for 2 <= x <= 1e6 (suffices at primes); pi(x) < Li(x) for 10 <= x <= 1e6
li2 = float(mpmath.li(2))
assert abs(expi(log(10.0)) - float(mpmath.li(10))) < 1e-9
lip = expi(np.log(P.astype(float)))
cntp = np.arange(1, len(P) + 1)
assert np.all(cntp < lip)
big = P >= 11
assert np.all(cntp[big] < lip[big] - li2) and 4 < Lis[10]
Li7 = float(mpmath.li(7, offset=True))
assert abs(Li7 - 3.71) < 0.005 and Li7 < 4
# ratio pi/Li within 2% from 1e4 on, within 0.2% at 1e6 (figure caption)
xi = np.arange(10**4, 10**6 + 1)
Lix = expi(np.log(xi.astype(float))) - li2
pcx = np.searchsorted(P, xi, side="right")
assert np.all(pcx[:-1] / Lix[1:] > 0.98) and np.all(pcx / Lix < 1.02)
assert abs(78498 / Lis[10**6] - 1) < 0.002

# ---- Chebyshev's constants
A = 0.5 * log(2) + log(3) / 3 + log(5) / 5 - log(30) / 30
assert abs(A - 0.921) < 1e-3 and abs(6 * A / 5 - 1.106) < 1e-3

# ---- Chebyshev bound theta(2n)-theta(n) < n log 4, n up to 5e5
th = np.zeros(10**6 + 1)
th[P] = np.log(P)
theta = np.cumsum(th)
nn = np.arange(1, 5 * 10**5 + 1)
assert np.all(theta[2 * nn] - theta[nn] < nn * log(4))
xs = np.arange(1, 10**6 + 1)
assert np.all(theta[xs] < 4 * log(2) * xs)
assert abs(theta[10] - log(210)) < 1e-12
assert 11 * 13 * 17 * 19 == 46189 and comb(20, 10) == 184756 and comb(20, 10) == 4 * 46189
assert abs(log(46189) - 10.7405) < 1e-4 and abs(10 * log(4) - 13.8629) < 1e-4
assert abs(8 * log(2) - 5.55) < 0.005
bound = 1000 + 8 * log(2) * 1e6 / log(1e6)
assert abs(bound - 402373) < 50 and 78498 < bound
assert abs(4 * log(2) - 2.77) < 0.005
# corollary O-statement: log x <= sqrt x, max 2/e at e^2
assert 2 / math.e < 1 and abs(log(math.e ** 2) / math.e - 2 / math.e) < 1e-12
xx = np.linspace(1.0001, 1e8, 200001)
assert np.all(np.log(xx) <= np.sqrt(xx))
assert np.all(np.sqrt(xx) + 8 * log(2) * xx / np.log(xx) <= (1 + 8 * log(2)) * xx / np.log(xx) * (1 + 1e-12))

# ---- Lambda table and identity
vals = [Lam(k) for k in range(1, 13)]
exp = [0, log(2), log(3), log(2), log(5), 0, log(7), log(2), log(3), 0, log(11), 0]
assert np.allclose(vals, exp)
for v, s in zip([0.693, 1.099, 1.609, 1.946, 2.398], [log(2), log(3), log(5), log(7), log(11)]):
    assert abs(v - s) < 5e-4
for n_ in range(1, 2001):
    assert abs(sum(Lam(d) for d in range(1, n_ + 1) if n_ % d == 0) - log(n_)) < 1e-9
assert abs(log(12) - 2.4849) < 1e-4
pp30 = [k for k in range(2, 31) if Lam(k) > 0]
assert pp30 == [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29]

# ---- psi
X = 10**5
L = np.zeros(X + 1)
for p in P[P <= X]:
    q = int(p)
    while q <= X:
        L[q] = log(p)
        q *= int(p)
psi = np.cumsum(L)
assert abs(psi[10] - log(2520)) < 1e-12 and abs(psi[10] - 7.832) < 1e-3
lcm10 = reduce(math.lcm, range(1, 11))
assert lcm10 == 2520
psi_tab = {10: (7.832, 0.783, -2.168), 100: (94.045, 0.940, -5.955),
           1000: (996.681, 0.997, -3.319), 10000: (10013.397, 1.001, 13.397)}
for x, (v, r, d) in psi_tab.items():
    assert abs(psi[x] - v) < 1e-3
    assert abs(psi[x] / x - r) < 1e-3
    assert abs(psi[x] - x - d) < 1e-3
for x in [10, 100, 1000, 12345, 99999]:
    s = sum(theta[int(math.floor(x ** (1.0 / m) + 1e-9))] for m in range(1, 40))
    assert abs(s - psi[x]) < 1e-8
for N in range(1, 61):
    assert abs(psi[N] - log(reduce(math.lcm, range(1, N + 1)))) < 1e-9
# Sample psi(30)
assert abs(psi[30] - 28.4765) < 1e-4 and abs(psi[30] / 30 - 0.949) < 1e-3
terms30 = [2.7726, 3.2958, 3.2189, 1.9459, 2.3979, 2.5649, 2.8332, 2.9444, 3.1355, 3.3673]
for v, s in zip(terms30,
                [4 * log(2), 3 * log(3), 2 * log(5), log(7), log(11), log(13), log(17), log(19), log(23), log(29)]):
    assert abs(v - s) < 1e-4
assert abs(sum(terms30) - 28.4764) < 1e-9
assert abs(psi[16] - psi[15] - log(2)) < 1e-12 and abs(psi[18] - psi[17]) < 1e-12
for x, rhs in [(100, 115.13), (1000, 1160.50), (10000, 11319.51)]:
    assert abs(pi_of(x) * log(x) - rhs) < 0.01 and psi[x] <= pi_of(x) * log(x)
cnt = np.cumsum(np.isin(np.arange(X + 1), P))
xs2 = np.arange(2, X + 1)
assert np.all(psi[xs2] <= cnt[xs2] * np.log(xs2) + 1e-9)

# ---- log p section: frequencies and periods
for q, lq, per in [(2, 0.693147, 9.0647), (3, 1.098612, 5.7192), (4, 1.386294, 4.5324),
                   (5, 1.609438, 3.9040), (7, 1.945910, 3.2289), (8, 2.079442, 3.0216),
                   (9, 2.197225, 2.8596)]:
    assert abs(log(q) - lq) < 1e-6 and abs(2 * pi / log(q) - per) < 1e-4
assert abs(2 * pi / log(2) - 9.065) < 1e-3
assert abs(1 / (2 * 2) - 0.25) < 1e-12 and abs(1 / sqrt(13) - 0.277) < 1e-3
for p in [2, 3, 5, 7]:
    for m in range(1, 6):
        assert abs(Lam(p ** m) / log(p ** m) - 1 / m) < 1e-12
# spectrum caption: six-prime terms with q <= 200
six = [p ** m for p in [2, 3, 5, 7, 11, 13] for m in (1, 2, 3)]
assert len(six) == 18 and len([q for q in six if q <= 200]) == 15
assert sorted(q for q in six if q > 200) == [343, 1331, 2197]

# ---- Paper connection
assert abs(1 / sqrt(2) - 0.707) < 1e-3
assert int(P[5]) == 13 and int(P[24]) == 97 and int(P[149]) == 863 and int(P[299]) == 1987
assert abs(3 * log(1987) - 22.78) < 0.01
assert [3 * k for k in [1, 3, 6, 25]] == [3, 9, 18, 75]
assert 3 * 300 == 900 and 3 * 25 * 2 == 150 and 3 * 150 * 2 == 900 and 450 * 2 == 900 and 3 * 24 == 72
assert pi_of(113) == 30
pp113 = [k for k in range(2, 114) if Lam(k) > 0]
higher = [k for k in pp113 if not isprime(k)]
assert sorted(higher) == [4, 8, 9, 16, 25, 27, 32, 49, 64, 81] and len(pp113) == 40
assert len(pp113) - 3 + 6 == 43
assert len([p for p in range(131, 212) if isprime(p)]) == 16
assert abs(1 / 16 - 0.0625) < 1e-12
assert abs(1 / (5 * 2 ** 2.5) - 0.035) < 5e-4 and abs(1 / (6 * 8) - 0.021) < 5e-4
assert abs(1 / (4 * 9) - 0.028) < 5e-4 and abs(1 / sqrt(211) - 0.069) < 5e-4
pp3000 = [k for k in range(2, 3001) if Lam(k) > 0]
assert pi_of(3000) == 430 and len(pp3000) == 466
assert abs(math.exp(8) - 2981) < 0.1
assert abs(log(1000) / 1000 - 0.007) < 2e-4
# training window length T for zeros 1-800
z800 = qmnt.zeros(800)
Tw = z800[-1] - z800[0]
assert abs(Tw - 1170) < 5 and abs(pi / Tw - 0.0027) < 1e-4 and abs(2 * pi / Tw - 0.0054) < 1e-4

# Table V row 401-500 via qmnt (null and oracle with first-k primes)
nh = np.arange(401, 501)
r0, _ = qmnt.score(qmnt.smooth_zeros(nh), nh)
assert abs(r0 - 0.263) < 0.002
for k, (rr_, cc_) in {1: (0.198, 0.66), 3: (0.116, 0.90), 6: (0.074, 0.96), 25: (0.043, 0.99)}.items():
    rk, ck = qmnt.score(qmnt.oracle_zeros(nh, qmnt.first_primes(k), mmax=3), nh)
    assert abs(rk - rr_) < 0.002 and abs(ck - cc_) < 0.01

# ---- Problems
assert 91 == 7 * 13 and isprime(97) and 119 == 7 * 17 and isprime(127) and 133 == 7 * 19
assert sqrt(97) < 10 and sqrt(127) < 11.3
assert factor(1001) == {7: 1, 11: 1, 13: 1} and factor(1024) == {2: 10}
assert factor(2310) == {2: 1, 3: 1, 5: 1, 7: 1, 11: 1} and factor(9999) == {3: 2, 11: 1, 101: 1}
assert Lam(1) == 0 and abs(Lam(27) - 1.0986) < 1e-4 and Lam(28) == 0
assert abs(Lam(29) - 3.3673) < 1e-4 and abs(Lam(32) - 0.6931) < 1e-4 and abs(Lam(121) - 2.3979) < 1e-4
assert abs(log(17) - 2.833) < 1e-3 and Lam(18) == 0 and Lam(16) > 0 and isprime(17)
assert int(P[999]) == 7919
assert abs(1000 * log(1000) - 6907.76) < 0.01 and abs(7919 / (1000 * log(1000)) - 1.146) < 1e-3
assert abs(1000 * (log(1000) + log(log(1000)) - 1) - 7840.4) < 0.1
assert abs(log(log(1000)) - 1.9326) < 1e-4
assert reduce(math.lcm, range(1, 13)) == 27720 and abs(psi[12] - log(27720)) < 1e-12
assert abs(log(27720) - 10.23) < 0.005
tw = [(p, p + 2) for p in range(2, 99) if isprime(p) and isprime(p + 2)]
assert len(tw) == 8
assert abs(2 * pi / log(11) - 2.620) < 1e-3 and abs(2 * pi / log(13) - 2.450) < 1e-3
assert abs(2 * pi / log(16) - 2.266) < 1e-3 and abs(2 * pi / log(97) - 1.373) < 1e-3
assert abs(2 * pi / (3 * log(97)) - 0.458) < 1e-3
assert abs(psi[100] / 100 - 0.9405) < 1e-4 and abs(psi[1000] / 1000 - 0.9967) < 1e-4
assert abs(psi[10000] / 10000 - 1.0013) < 1e-4
for nn_ in range(1, 300):
    assert comb(2 * nn_, nn_) >= 4 ** nn_ / (2 * nn_ + 1)
    assert comb(2 * nn_, nn_) <= (2 * nn_) ** pi_of(2 * nn_)
    assert max(comb(2 * nn_, k) for k in range(2 * nn_ + 1)) == comb(2 * nn_, nn_)
bnd50 = (50 * log(4) - log(101)) / log(100)
assert abs(bnd50 - 14.05) < 0.005 and bnd50 <= 25
assert abs(50 * log(4) - 69.315) < 1e-3 and abs(log(101) - 4.615) < 1e-3
x = 1e6
t1 = x / log(x)
t2 = t1 + x / log(x) ** 2
t3 = t2 + 2 * x / log(x) ** 3
assert abs(t1 - 72382.4) < 0.1 and abs(t2 - 77621.7) < 0.1 and abs(t3 - 78380.1) < 0.1
C = -2 / log(2) - 2 / log(2) ** 2
I3 = float(mpmath.quad(lambda t: 1 / mpmath.log(t) ** 3, [2, 100, 10**4]))
assert abs(1e4 / log(1e4) + 1e4 / log(1e4) ** 2 + 2 * I3 + C - Lis[10**4]) < 1e-6
tz = qmnt.zeros(10)
RE1 = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
       40.918719, 43.327073, 48.005151, 49.773832]
assert np.allclose(tz, RE1, atol=1e-6)
a = np.array([qmnt.Nfl(t, [2], mmax=3) for t in tz])
b = np.array([qmnt.Nfl(t + 2 * pi / log(2), [2], mmax=3) for t in tz])
assert np.max(np.abs(a - b)) < 1e-10
direct = np.array([-sum(math.sin(m * t * log(2)) / (m * 2 ** (m / 2)) for m in (1, 2, 3)) / pi for t in tz])
assert np.allclose(a, direct, atol=1e-12)
assert abs(1 / (3 * 2 ** 1.5) - 1 / (6 * sqrt(2))) < 1e-12
eprime = lambda e: e % 2 == 0 and e % 4 != 0
assert all(eprime(e) for e in [2, 6, 10, 30]) and 2 * 30 == 60 == 6 * 10
prod30 = 17 * 19 * 23 * 29
assert prod30 == 215441 and comb(30, 15) % prod30 == 0 and comb(30, 15) // prod30 == 720
assert abs(log(1e6) - 13.8) < 0.05 and abs(1e6 / 78498 - 12.74) < 0.005

print("All checks passed.")
