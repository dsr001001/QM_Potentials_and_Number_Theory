import numpy as np
from math import comb, ceil, sqrt, log, exp, pi
from itertools import product, combinations
from scipy import integrate, stats
import qmnt

# Sample 1: SE of small sample
x = np.array([3., 7, 4, 9, 2])
s = x.std(ddof=1)
assert abs(x.mean() - 5) < 1e-12 and abs(s - 2.9155) < 1e-3
assert abs(s/np.sqrt(5) - 1.3038) < 1e-3 and abs(x.var() - 6.8) < 1e-12
assert abs(np.sqrt(6.8/5) - 1.166) < 1e-3          # ideal bootstrap SE
rng = np.random.default_rng(0)
m = x[rng.integers(0, 5, (20000, 5))].mean(1)
assert abs(m.std() - 1.166) < 0.02
B = 20000
assert abs(0.5*np.sqrt(2/(B - 1)) - 0.005) < 1e-5 and abs(1/np.sqrt(2*B) - 0.005) < 1e-6

# Sample 2: enumeration of (1,2,6)
y = np.array([1., 2, 6])
sums = np.array([y[list(i)].sum() for i in product(range(3), repeat=3)])
means = sums/3
assert abs(sums.sum() - 243) < 1e-9 and abs(sums.sum()/(27*3) - 3) < 1e-12
assert abs((sums**2).sum() - 2565) < 1e-9 and abs((sums**2).sum()/9 - 285) < 1e-9
assert abs(means.mean() - 3) < 1e-12
assert abs((means**2).mean() - 285/27) < 1e-12
assert abs(means.var() - 14/9) < 1e-12 and abs(np.sqrt(14/9) - 1.247) < 1e-3
assert len(set(np.round(means, 10))) == 10 == comb(5, 3)
assert abs(comb(199, 100) / 4.5e58 - 1) < 0.02
def q(qq):
    v = np.sort(means); F = np.arange(1, 28)/27
    return v[np.argmax(F >= qq - 1e-12)]
assert abs(q(0.10) - 4/3) < 1e-12 and abs(q(0.90) - 14/3) < 1e-12
assert abs(q(0.025) - 1) < 1e-12 and abs(q(0.975) - 6) < 1e-12

# Sample 4: oracle bootstrap; null anchor
n = np.arange(401, 501)
t = qmnt.zeros(500)[n - 1]
tp = qmnt.oracle_zeros(n, qmnt.first_primes(6))
r_, c_ = qmnt.score(tp, n)
assert abs(r_ - 0.074) < 0.0006 and abs(c_ - 0.96) < 0.006
rn, _ = qmnt.score(qmnt.smooth_zeros(n), n)
assert abs(rn - 0.2631) < 0.0006                 # null, Problem 21
ft, fp = qmnt.fluct(t, n), qmnt.fluct(tp, n)
rng = np.random.default_rng(1)
idx = rng.integers(0, 100, (20000, 100))
a = ft[idx] - ft[idx].mean(1, keepdims=True); b = fp[idx] - fp[idx].mean(1, keepdims=True)
rr = (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
lo, hi = np.percentile(rr, [2.5, 97.5])
assert abs(lo - 0.95) < 0.01 and abs(hi - 0.97) < 0.01

# Sample 5: AR(1)
def fn(phi, N):
    k = np.arange(1, N)
    return 1 + 2*np.sum((1 - k/N)*phi**k)
assert abs(fn(0.5, 200) - 2.98) < 0.005 and abs(200/fn(0.5, 200) - 67.1) < 0.2
assert abs(np.sqrt(fn(0.5, 200)) - 1.726) < 0.002
assert abs(fn(-0.5, 200) - 0.3356) < 0.001 and abs(200/fn(-0.5, 200) - 596) < 1.5
assert abs(np.sqrt(fn(-0.5, 200)) - 0.579) < 0.002
for phi in (0.5, -0.5):
    N = 10
    approx = 1 + 2*(phi/(1 - phi) - phi/(1 - phi)**2/N)
    bound = abs(phi)**(N + 1)/(N*(1 - abs(phi))**2)
    assert abs(fn(phi, N) - approx) <= 2*bound*(1 + 1e-9) + 1e-14
f400 = fn(0.5, 400); fb20 = fn(0.5, 20)
assert abs(f400 - 2.99) < 0.005 and abs(fb20 - 2.8) < 0.005
assert abs(1/np.sqrt(f400) - 0.578) < 0.002 and abs(1 - np.sqrt(fb20/f400) - 0.03) < 0.01

# Problem 7 (new parameters): phi=-0.3, n=100
ph7 = -0.3
f7 = 1 + 2*(ph7/(1 - ph7) - ph7/(1 - ph7)**2/100)
assert abs(ph7/(1 - ph7) + 0.23077) < 1e-5 and abs(ph7/(1 - ph7)**2 + 0.17751) < 1e-5
assert abs(f7 - 0.5420) < 1e-4 and abs(100/f7 - 184.5) < 0.1
assert abs(fn(ph7, 100) - f7) < 1e-6
finf7 = 0.7/1.3
assert abs(finf7 - 0.5385) < 1e-4 and (f7 - finf7)/finf7 < 0.01

# block length rule of thumb
assert abs(100**(1/3) - 4.64) < 0.01

# Sample 6: block counts
cnt = lambda i: min(i, 96) - max(1, i - 4) + 1
assert (cnt(1), cnt(3), cnt(50)) == (1, 3, 5)
assert abs(20/96 - 0.208) < 1e-3 and abs(100/96 - 1.042) < 1e-3

# Sample 7: permutation test
pool = [1, 2, 3, 4, 6, 8]
D = [(2*sum(c) - 24)/3 for c in combinations(pool, 3)]
assert sum(d >= 10/3 - 1e-12 for d in D)/20 == 0.1
assert sum(abs(d) >= 10/3 - 1e-12 for d in D)/20 == 0.2
assert abs(1/201 - 0.004975) < 1e-6 and abs(1/101 - 0.0099) < 1e-4
assert abs(4/100 - 0.04) < 1e-12
qq = 1 - exp(log(0.05)/200); assert abs(qq - 0.01487) < 1e-4

# Sample 9: phase randomization
d = np.array([1., 2, 0, -1]); Dd = np.fft.fft(d)
assert np.allclose(Dd, [2, 1-3j, 0, 1+3j])
D2 = Dd.copy(); D2[1] *= np.exp(1j*pi/4); D2[3] *= np.exp(-1j*pi/4)
dp = np.fft.ifft(D2).real
assert np.allclose(dp, [1.914, 1.207, -0.914, -0.207], atol=1e-3)
cc = lambda v, k: np.sum(v*np.roll(v, -k))
for k, val in zip(range(3), [6, 1, -4]):
    assert abs(cc(d, k) - val) < 1e-12 and abs(cc(dp, k) - val) < 1e-12
# centered circular autocorrelation preserved
md = d.mean()
for k in range(4):
    assert abs((cc(d, k) - 4*md**2) - (cc(dp, k) - 4*dp.mean()**2)) < 1e-12

# Sample 10: paper surrogate numbers
for m_, sd, z in [(0.134, 0.010, 0.3), (0.141, 0.006, -0.67), (0.138, 0.005, -0.2)]:
    assert abs((0.137 - m_)/sd - z) < 0.01
for v, f in [(0.137, 0.439), (0.134, 0.451), (0.141, 0.422), (0.138, 0.434)]:
    assert abs(1 - v/0.244 - f) < 1e-3

# Toy demonstration (Sec. capacity): RE2 target on levels 101--200
nn = np.arange(101, 201)
sn = qmnt.smooth_zeros(nn)
tgt = -qmnt.Nfl(sn, np.array([2]), mmax=3)
trms = np.sqrt(np.mean(tgt**2))
jj = np.arange(100)
def gen_res(yv, M):
    if M == 0:
        return np.sqrt(np.mean(yv**2))
    C = np.cos(np.pi*np.arange(M)[None, :]*(jj[:, None] + 0.5)/100)
    cf = np.linalg.lstsq(C, yv, rcond=None)[0]
    return np.sqrt(np.mean((yv - C @ cf)**2))
S = np.column_stack([f(mm*sn*np.log(2)) for mm in (1, 2, 3) for f in (np.sin, np.cos)])
def str_res(yv):
    cf = np.linalg.lstsq(S, yv, rcond=None)[0]
    return np.sqrt(np.mean((yv - S @ cf)**2))
assert str_res(tgt) < 1e-10
def phr(dv, rg):
    Dv = np.fft.rfft(dv); ph = rg.uniform(0, 2*np.pi, len(Dv)); ph[0] = 0.0
    if len(dv) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(Dv*np.exp(1j*ph), n=len(dv))
rg = np.random.default_rng(7)
phs = [phr(tgt, rg) for _ in range(20)]
pms = [rg.permutation(tgt) for _ in range(20)]
Ms = list(range(0, 101, 10))
real_curve = np.array([gen_res(tgt, M) for M in Ms])/trms
ph_curve = np.mean([[gen_res(yy, M) for M in Ms] for yy in phs], axis=0)/trms
assert np.max(np.abs(real_curve - ph_curve)) < 0.2
assert gen_res(tgt, 100) < 1e-10 and max(gen_res(yy, 100) for yy in phs + pms) < 1e-10
for fam in (phs, pms):
    assert np.mean([str_res(yy)/trms for yy in fam]) > 0.5

# Sample 11: multiple looks; Bonferroni; Problem 14
assert abs(1 - 0.95**5 - 0.2262) < 1e-4 and abs(1 - 0.95**20 - 0.6415) < 1e-4
assert abs(1 - 0.9975**20 - 0.0488) < 1e-4 and abs(1 - 0.95**10 - 0.4013) < 1e-4
assert (1 - 0.05/20)**20 >= 0.95

# Sample 12: chance matches; Problem 29
z = qmnt.zeros(400); T = z[399] - z[0]
assert abs(T - 666) < 3
w = 2*pi/666; assert abs(w - 0.00943) < 1e-5
la = 13/log(13)*w; lb = exp(8)/8*w; lc = 101/log(101)*w
assert abs(log(13) - 2.565) < 1e-3 and abs(log(101) - 4.615) < 1e-3
assert abs(la - 0.048) < 0.001 and abs(1 - exp(-la) - 0.047) < 0.001
assert abs(lb - 3.5) < 0.05 and abs(1 - exp(-lb) - 0.970) < 0.002
assert abs(lc - 0.21) < 0.005 and abs(1 - exp(-lc) - 0.19) < 0.01
# prime-power density near 13 vs 1/log estimate: within a factor of about two
pr = qmnt.primes_upto(100)
pp = sorted({int(p)**k for p in pr for k in range(1, 8) if int(p)**k <= 100})
near = [qv for qv in pp if abs(log(qv) - log(13)) < 0.4]
ratio = (len(near)/0.8)/(13/log(13))
assert 0.5 < ratio < 2.0

# d_R constants and Sample 13
def dR(R):
    f = lambda u: 1 - stats.norm.cdf(u)**R - (1 - stats.norm.cdf(u))**R
    return integrate.quad(f, -10, 10)[0]
for R, v in [(2, 1.128), (4, 2.059), (10, 3.078)]:
    assert abs(dR(R) - v) < 1e-3
assert abs(0.014/2.059 - 0.0068) < 1e-4 and abs(0.10/3.92 - 0.0255) < 1e-4
assert abs(0.0255/0.0068 - 3.75) < 0.01
assert abs((0.43 - 0.33) - 0.10) < 1e-12 and abs((0.42 - 0.32) - 0.10) < 1e-12

# Problem answers 1, 2, 5, 13, 19, 26, 31
v = np.array([2., 5, 8]); assert abs(v.std(ddof=1)/sqrt(3) - 1.732) < 1e-3
assert abs(np.sqrt(v.var()/3) - 1.414) < 1e-3
assert comb(7, 4) == 35 and 4**4 == 256
assert abs(np.var([0, .5, .5, 1]) - 1/8) < 1e-12
assert abs(0.1483/0.137 - 1.082) < 1e-3 and abs(0.1493/0.137 - 1.090) < 1e-3
g0 = 4/3
assert abs(np.sqrt(g0*f400/400) - 0.0998) < 2e-4 and abs(np.sqrt(g0/400) - 0.0577) < 1e-4
assert abs(fn(0.5, 5) - 2.225) < 1e-9 and abs(np.sqrt(g0*2.225/400) - 0.0861) < 1e-4
assert abs(np.sqrt(g0*fb20/400) - 0.0966) < 2e-4
assert abs(0.010/sqrt(10) - 0.0032) < 1e-4
assert ceil(0.025*20000) == 500 and ceil(0.975*20000) == 19500

# AR(1) block-bootstrap simulation behind Fig. ar1 (qualitative claims)
rng = np.random.default_rng(5)
def ar1(phi, N):
    e = rng.normal(size=N + 200); xx = np.empty(N + 200); xx[0] = e[0]
    for j in range(1, N + 200):
        xx[j] = phi*xx[j-1] + e[j]
    return xx[200:]
true_se = np.sqrt(g0*f400/400)
def cbb_ratio(ell, nser=100, nres=300, N=400):
    out = []
    nb = -(-N // ell)
    for _ in range(nser):
        xs = ar1(0.5, N)
        st = rng.integers(0, N, size=(nres, nb))
        ii = ((st[:, :, None] + np.arange(ell)[None, None, :]) % N).reshape(nres, -1)[:, :N]
        out.append(xs[ii].mean(axis=1).std())
    return np.mean(out)/true_se
r1 = cbb_ratio(1); r20 = cbb_ratio(20)
assert abs(r1 - 0.578) < 0.05
assert 0.85 < r20 < 1.02 and r20 > r1
print('all checks passed')
