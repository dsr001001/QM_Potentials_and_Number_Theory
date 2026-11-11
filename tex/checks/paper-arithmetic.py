import numpy as np, qmnt
from scipy.optimize import brentq

# Sample "One oracle step by hand"
s = 680.0; l2 = np.log(2)
nfl = lambda t: -sum(np.sin(m*t*l2)/(m*2**(m/2)) for m in (1, 2, 3))/np.pi
dnfl = lambda t: -sum(l2*np.cos(m*t*l2)/2**(m/2) for m in (1, 2, 3))/np.pi
assert abs(nfl(s) + 0.0499) < 2e-4
sl = np.log(s/(2*np.pi))/(2*np.pi)
assert abs(sl - 0.7455) < 2e-4 and abs(1/sl - 1.3414) < 2e-4
da = -nfl(s)/sl; assert abs(da - 0.0670) < 1e-3
assert abs(dnfl(s) + 0.3377) < 1e-3
db = -nfl(s)/(sl + dnfl(s)); assert abs(db - 0.122) < 2e-3
g = 1/sl
G = lambda t: qmnt.Nsm(t) - qmnt.Nsm(s) + nfl(t)
dc = brentq(G, s - 3*g, s + 3*g) - s
assert abs(dc - 0.120) < 3e-3 and abs(dc/g - 0.090) < 3e-3
assert abs(0.5*g**2/(2*np.pi*680) - 2e-4) < 5e-5

# Sample "unique root"
def B(P): return sum(np.log(p)*(p**-0.5 + p**-1 + p**-1.5) for p in P)/np.pi
assert abs(B([2]) - 0.3443) < 1e-3
assert abs(2*np.pi*np.exp(2*np.pi*B([2])) - 54.7) < 0.3
P6 = [2, 3, 5, 7, 11, 13]
assert abs(B(P6) - 2.09) < 0.01
assert abs(2*np.pi*np.exp(2*np.pi*B(P6))/3.2e6 - 1) < 0.1
def rslope(P): return np.sqrt(sum(np.log(p)**2*sum(1/(2*p**m) for m in (1, 2, 3)) for p in P))/np.pi
r6 = rslope(P6)
assert abs(r6 - 0.415) < 0.01
tg = np.linspace(1000, 200000, 400001)
d6 = -sum(np.log(p)*np.cos(m*tg*np.log(p))/p**(m/2) for p in P6 for m in (1, 2, 3))/np.pi
assert abs(np.sqrt(np.mean(d6**2)) - r6) < 0.01
assert abs(0.7455/r6 - 1.8) < 0.05
assert abs(rslope(qmnt.first_primes(25)) - 0.7) < 0.06
assert abs(rslope(qmnt.first_primes(300)) - 1.2) < 0.08
assert B([2, 3]) > 0

# Table V row 1 rerun
n = np.arange(401, 501); tt = np.asarray(qmnt.zeros(500))[400:500]
assert abs(qmnt.rms(qmnt.fluct(tt, n)) - 0.263) < 1e-3
fl = np.asarray(qmnt.fluct(tt, n)); assert abs(np.mean(fl)) < 0.05
assert 0.05**2/0.263**2 < 0.04
for k, (r0, c0) in {1: (0.198, 0.66), 3: (0.116, 0.90), 6: (0.074, 0.96), 25: (0.043, 0.99)}.items():
    r, c = qmnt.score(qmnt.oracle_zeros(n, qmnt.first_primes(k), mmax=3), n)
    assert abs(r - r0) < 1.5e-3 and abs(c - c0) < 0.01
n2 = np.arange(1001, 2001)
r, c = qmnt.score(qmnt.oracle_zeros(n2, qmnt.first_primes(6), mmax=3), n2)
assert abs(r - 0.091) < 1.5e-3
# Problem 11: six primes on zeros 101-400
n3 = np.arange(101, 401)
r3, c3 = qmnt.score(qmnt.oracle_zeros(n3, qmnt.first_primes(6), mmax=3), n3)
assert r3 < 0.1 and c3 > 0.9

# Table ratios and problem 3
nulls = [0.263, 0.270, 0.279, 0.288, 0.296]
cols = {1: [0.198, 0.208, 0.219, 0.232, 0.242], 3: [0.116, 0.127, 0.139, 0.156, 0.169],
        6: [0.074, 0.083, 0.091, 0.107, 0.119], 25: [0.043, 0.050, 0.054, 0.060, 0.067]}
ratio = {1: [0.75, 0.77, 0.78, 0.81, 0.82], 3: [0.44, 0.47, 0.50, 0.54, 0.57],
         6: [0.28, 0.31, 0.33, 0.37, 0.40], 25: [0.16, 0.19, 0.19, 0.21, 0.23]}
for k in cols:
    for v, nl, rr in zip(cols[k], nulls, ratio[k]):
        assert abs(v/nl - rr) < 0.006
for v, nl, a in zip(cols[1], nulls, [0.247, 0.230, 0.215, 0.194, 0.182]):
    assert abs(1 - v/nl - a) < 1.5e-3
# ratio drift factors 1.1 to 1.4
for k, (lo, hi) in {1: (1.05, 1.15), 3: (1.25, 1.35), 6: (1.38, 1.45), 25: (1.35, 1.45)}.items():
    f = (cols[k][-1]/nulls[-1])/(cols[k][0]/nulls[0]); assert lo < f < hi
z10 = np.asarray(qmnt.zeros(10000)); assert abs(z10[-1] - 9877.78) < 0.01
assert abs(z10[399] - 680) < 0.5 and 14 < z10[-1]/z10[399] < 15

# factor 1.3
assert abs(0.265/0.198 - 1.34) < 0.01 and abs(0.279/0.208 - 1.34) < 0.01
assert abs(0.273/0.198 - 1.38) < 0.005 and abs(0.273/0.242 - 1.13) < 0.005

# rms bound sample, problem 7
assert abs(0.263*np.sqrt(1 - 0.28**2) - 0.252) < 1e-3
assert abs(np.sqrt(1 - (0.028/0.263)**2) - 0.994) < 1e-3
assert abs(0.263*np.sqrt(1 - 0.78**2) - 0.165) < 1e-3
rts = np.roots([1, -2*0.28*0.263, 0.263**2 - 0.258**2]).real
assert abs(min(rts) - 0.02) < 0.005 and abs(max(rts) - 0.13) < 0.005 and max(rts) <= 0.5*0.263

# parameter counts
pr = qmnt.first_primes(300)
assert pr[299] == 1987 and abs(3*np.log(1987) - 22.78) < 0.01
assert qmnt.first_primes(25)[24] == 97 and qmnt.first_primes(150)[149] == 863
assert abs(3*np.log(863) - 20.3) < 0.05

# C1 grid sample
z = np.asarray(qmnt.zeros(800)); T = z[399] - z[0]
assert abs(T - 666) < 2 and abs(2*np.pi/T - 0.0094) < 1e-4
T8 = z[799] - z[0]
assert abs(T8 - 1170) < 3 and abs(2*np.pi/T8 - 0.0054) < 1e-4 and abs(np.pi/T8 - 0.0027) < 1e-4
for q, j0, dr in ((2, 39, 0.12), (3, 62, 2.4), (5, 91, 5.6)):
    j = round(np.log(q)*450/8); assert j == j0
    assert abs(abs(8*j/450 - np.log(q))*666 - dr) < 0.06*max(1, dr)
assert abs(8/450 - 0.0178) < 1e-4
assert abs(np.pi/(2*T) - 0.00236) < 2e-5
assert abs((8 - 0.05)/(np.pi/(2*T)) - 3370) < 20
assert 133 == 400//3 and 266 == 800//3

# 43 prime powers and weights
ps = list(qmnt.primes_upto(113)); assert len(ps) == 30
hp = [q for q in range(2, 114) if any(q == p**m for p in ps for m in range(2, 8))]
assert len(hp) == 10 and 30 + 10 - 3 + 6 == 43
assert abs(1/(5*2**2.5) - 0.0354) < 1e-4 and abs(1/48 - 0.0208) < 1e-4
assert abs(1/36 - 0.0278) < 1e-4 and abs(1/np.sqrt(211) - 0.0688) < 1e-4
assert abs(1/np.sqrt(113) - 0.0941) < 1e-4 and abs(1/16 - 0.0625) < 1e-9
assert abs(1/(3*np.sqrt(27)) - 0.0642) < 1e-4 and abs(1/14 - 0.0714) < 1e-4
assert len([p for p in qmnt.primes_upto(211) if p >= 127]) == 17
assert abs(np.log(139/137) - 0.0145) < 1e-4 and abs(np.log(128/127) - 0.0078) < 1e-4
assert abs(0.0145/0.0054 - 2.7) < 0.05 and abs(0.0078/0.0054 - 1.5) < 0.06

# chance match sample, problem 9
for w, lam, pp in ((2, 0.035, 0.034), (np.log(113), 0.23, 0.20), (8, 3.5, 0.97)):
    L = 0.0094*np.exp(w)/w
    assert abs(L - lam) < 0.05*lam + 1e-3 and abs(1 - np.exp(-L) - pp) < 0.02
assert abs(0.0054*211/np.log(211) - 0.21) < 0.01
assert abs(1/201 - 0.005) < 1e-3

# dephasing
assert abs(1/(np.pi*np.sqrt(2)) - 0.225) < 1e-3
assert abs(np.sin(0.02)/np.pi - 0.0064) < 1e-4 and abs(np.sin(0.08)/np.pi - 0.025) < 5e-4
assert abs(z[399]/2 + z[0]/2 - 347) < 1
assert abs(0.0047*1600 - 7.5) < 0.05
assert abs(2e-3*(2000 - 347) - 3.3) < 0.05
assert abs(np.asarray(qmnt.zeros(2000))[1999] - 2515) < 3

# root ambiguity, problem 21
assert abs(0.7455/5 - 0.149) < 1e-3
assert abs(-0.7455/1.5 + 0.497) < 1e-3

# problem 26: prime powers below e^2
pp2 = [q for q in range(2, 8) if any(q == p**m for p in (2, 3, 5, 7) for m in range(1, 4))]
assert pp2 == [2, 3, 4, 5, 7] and abs(np.exp(2) - 7.39) < 0.01

# C2 first five frequencies match prime powers
t4 = z[:400]; y = np.arange(1, 401) - 0.5 - qmnt.Nsm(t4)
d = 0.25*2*np.pi/T; coarse = np.arange(0.05, 8.0, d)
pk = lambda r, gr: gr[np.argmax(np.abs(np.exp(-1j*np.outer(gr, t4)) @ r)**2)]
lq = np.array(sorted({m*np.log(p) for p in qmnt.primes_upto(3000) for m in range(1, 12) if p**m <= 3000}))
fr, r = [], y.copy()
for _ in range(5):
    w0 = pk(r, coarse); fr.append(pk(r, np.linspace(w0 - d, w0 + d, 201)))
    X = np.hstack([np.sin(np.outer(t4, fr)), np.cos(np.outer(t4, fr))])
    cf, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ cf
assert abs(fr[0] - np.log(2)) < np.pi/T
for w in fr:
    assert np.min(np.abs(lq - w)) < np.pi/T

# Problem 15: periodogram peak-to-median ratio, real vs permuted (seed 1)
Em = np.exp(-1j*np.outer(coarse, t4))
Pw = np.abs(Em @ y)**2
yp = np.random.default_rng(1).permutation(y)
Pp = np.abs(Em @ yp)**2
assert Pw.max()/np.median(Pw) > 3*Pp.max()/np.median(Pp)

# Control B 25 primes (simplified validation)
P = qmnt.first_primes(25)
w = np.array([m*np.log(p) for p in P for m in (1, 2, 3)])
t5 = np.asarray(qmnt.zeros(500)); y5 = np.arange(1, 501) - 0.5 - qmnt.Nsm(t5)
Xf = lambda tt_: np.hstack([np.sin(np.outer(tt_, w)), np.cos(np.outer(tt_, w))])
def ridge(Xm, yv, lam):
    U, sv, Vt = np.linalg.svd(Xm, full_matrices=False)
    return Vt.T @ ((sv/(sv**2 + lam*sv[0]**2))*(U.T @ yv))
lams = 10.0**np.arange(-8, 3.01, 0.5); assert len(lams) == 23
val = [np.sqrt(np.mean((Xf(t5[300:400]) @ ridge(Xf(t5[:300]), y5[:300], L) - y5[300:400])**2)) for L in lams]
th = ridge(Xf(t5[:400]), y5[:400], lams[int(np.argmin(val))])
Ff = lambda tt_: float(np.ravel(Xf(np.atleast_1d(tt_)) @ th)[0])
sa, ga = np.ravel(qmnt.smooth_zeros(n)), np.ravel(qmnt.mean_gap(n))
def pred(k, s0, g0):
    Gk = lambda tt_: float(np.ravel(qmnt.Nsm(tt_))[0]) + Ff(tt_) - (k - 0.5)
    a, b = s0 - 3*g0, s0 + 3*g0
    while Gk(a)*Gk(b) > 0: a, b = a - g0, b + g0
    return brentq(Gk, a, b)
tp = np.array([pred(int(k), float(s0), float(g0)) for k, s0, g0 in zip(n, sa, ga)])
rB, cB = qmnt.score(tp, n)
assert rB < 0.074 and cB > 0.95
assert abs(th[0] + 0.225) < 0.05 and abs(th[75]) < 0.05

# misc
assert abs(0.180 - 0.012 - 0.168) < 1e-9 and 0.192 < 0.263
assert abs(1170/666 - 1.76) < 0.01
print("all checks passed")
