import numpy as np, math, itertools
from scipy import special, stats, integrate
import qmnt

# ---------- Sample Problem: fair die ----------
vals = np.arange(1, 7); p = np.ones(6) / 6
mu = np.sum(vals * p); EX2 = np.sum(vals**2 * p); var = EX2 - mu**2
assert abs(mu - 3.5) < 1e-12
assert abs(EX2 - 91/6) < 1e-12
assert abs(var - 35/12) < 1e-12 and abs(var - 2.9167) < 1e-4
assert abs(math.sqrt(var) - 1.708) < 1e-3

# ---------- Uniform [0,1]: variance 1/12, sd 0.2887 ----------
assert abs(1/3 - 1/4 - 1/12) < 1e-15
assert abs(1/math.sqrt(12) - 0.2887) < 1e-4

# ---------- Sample Problem: RE1 spacings ----------
t = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
assert np.allclose(qmnt.zeros(10), t, atol=1e-6)
gaps = np.diff(t)
assert np.allclose(gaps, [6.887315, 3.988818, 5.414018, 2.510186, 4.651116,
                          3.332541, 2.408354, 4.678078, 1.768681], atol=1e-6)
assert abs(t[-1] - t[0] - 35.639107) < 1e-6
assert abs(gaps.mean() - 3.959901) < 1e-6
Q = np.sum((gaps - gaps.mean())**2)
assert abs(Q - 21.3826) < 1e-3
assert abs(Q/9 - 2.37584) < 1e-4 and abs(Q/8 - 2.67282) < 1e-4
assert abs(gaps.std(ddof=0) - 1.5414) < 1e-3
assert abs(gaps.std(ddof=1) - 1.6349) < 1e-3

# ---------- Sample Problem: bias, scatter, rms ----------
e = np.array([0.1, 0.3, 0.2])
assert abs(e.mean() - 0.2) < 1e-12
assert abs(np.mean(e**2) - 0.046667) < 1e-6
assert abs(np.sqrt(np.mean(e**2)) - 0.21602) < 1e-5
assert abs(e.std() - 0.08165) < 1e-5
assert abs(e.mean()**2 + e.var() - np.mean(e**2)) < 1e-14

# ---------- Gaussian table ----------
assert abs(special.erf(1/math.sqrt(2)) - 0.6827) < 1e-4
assert abs(special.erf(2/math.sqrt(2)) - 0.9545) < 1e-4
assert abs(special.erf(3/math.sqrt(2)) - 0.9973) < 1e-4
assert abs(stats.norm.ppf(0.975) - 1.960) < 1e-3
assert abs(special.erf(1.96/math.sqrt(2)) - 0.95) < 1e-3
I0 = integrate.quad(lambda u: np.exp(-u**2/2), -np.inf, np.inf)[0]
assert abs(I0 - math.sqrt(2*math.pi)) < 1e-9
I2 = integrate.quad(lambda u: u**2*np.exp(-u**2/2), -np.inf, np.inf)[0]
assert abs(I2 - math.sqrt(2*math.pi)) < 1e-9

# ---------- Poisson limit table ----------
for nn, val in [(10, 0.348678), (100, 0.366032), (1000, 0.367695)]:
    assert abs((1 - 1/nn)**nn - val) < 1e-6
assert abs(math.exp(-1) - 0.367879) < 1e-6
s_, n_ = 1.0, 100
rem = -n_*math.log(1 - s_/n_) - s_
assert 0 <= rem <= s_**2/(n_ - s_)

# ---------- Sample Problem: gap probabilities ----------
assert abs((1 - math.exp(-0.1)) - 0.0952) < 1e-4
assert abs(math.exp(-0.1) - 0.904837) < 1e-6
assert abs(math.exp(-2) - 0.1353) < 1e-4
assert abs(math.log(2) - 0.693147) < 1e-6
m1 = integrate.quad(lambda s: s*np.exp(-s), 0, np.inf)[0]
m2 = integrate.quad(lambda s: s**2*np.exp(-s), 0, np.inf)[0]
assert abs(m1 - 1) < 1e-9 and abs(m2 - 2) < 1e-9   # Problem 16

# ---------- Sample Problem: correlation by hand ----------
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 1, 4, 3, 5.])
assert abs(np.corrcoef(x, y)[0, 1] - 0.8) < 1e-12

# ---------- Sample Problem: correct shape, wrong amplitude ----------
yy = np.array([0.3, -0.2, 0.1, -0.4, 0.2])
assert abs(yy.mean()) < 1e-12
rmsy = np.sqrt(np.mean(yy**2))
assert abs(np.mean(yy**2) - 0.068) < 1e-12
assert abs(rmsy - 0.2608) < 1e-4
for a, claimed in [(0.5, 0.1304), (2, 0.2608), (3, 0.5215)]:
    assert abs(np.sqrt(np.mean((a*yy - yy)**2)) - claimed) < 1e-4
    assert abs(np.corrcoef(a*yy, yy)[0, 1] - 1) < 1e-12

# ---------- identity Eq. msedecomp (numerical instance) ----------
rng = np.random.default_rng(123)
yt = rng.normal(0.1, 1, 50); pp = 0.3*yt + rng.normal(0.2, 0.5, 50)
lhs = np.mean((pp - yt)**2)
r = np.corrcoef(pp, yt)[0, 1]
rhs = (pp.mean()-yt.mean())**2 + pp.var() + yt.var() - 2*r*pp.std()*yt.std()
assert abs(lhs - rhs) < 1e-12

# ---------- Sample Problem: best rms at correlation r (Table V) ----------
assert abs(math.sqrt(1 - 0.66**2) - 0.7513) < 1e-4
assert abs(0.263*math.sqrt(1 - 0.66**2) - 0.1976) < 1e-4
assert abs(math.sqrt(0.19) - 0.4359) < 1e-4
assert abs(0.263*math.sqrt(1 - 0.90**2) - 0.1146) < 1e-4
assert abs(0.263*math.sqrt(1 - 0.96**2) - 0.0736) < 1e-4
assert abs(0.263*math.sqrt(1 - 0.99**2) - 0.037) < 5e-4
assert abs(0.263*math.sqrt(1 - 0.985**2) - 0.045) < 5e-4
assert abs(math.sqrt(1 - 0.96**2) - 0.28) < 1e-12

# ---------- Sample Problem: alternating sequence ----------
def autocorr(x, kmax):
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x**2)
    return np.array([np.sum(x[:len(x)-k]*x[k:])/den for k in range(kmax+1)])
ra = autocorr([1, -1, 1, -1, 1, -1], 2)
assert abs(ra[1] + 5/6) < 1e-12 and abs(ra[2] - 4/6) < 1e-12

# ---------- Sample Problem: rms of RE2 ----------
a1 = 1/(math.pi*math.sqrt(2)); a2 = 1/(4*math.pi); a3 = 1/(6*math.sqrt(2)*math.pi)
assert abs(a1 - 0.225079) < 1e-6 and abs(a2 - 0.079577) < 1e-6 and abs(a3 - 0.037513) < 1e-6
assert abs(a1**2 - 0.050661) < 1e-6 and abs(a2**2 - 0.006333) < 1e-6 and abs(a3**2 - 0.001407) < 1e-6
ms = (a1**2 + a2**2 + a3**2)/2
assert abs(ms - 0.029200) < 1e-6
rms_re2 = math.sqrt(ms)
assert abs(rms_re2 - 0.17088) < 1e-5
# m=1 term carries 87% of the mean square (autocorrelation discussion)
assert abs(round(100*a1**2/(a1**2 + a2**2 + a3**2)) - 87) < 1e-12
# sampled at smooth zeros 401-500: close to the paper's probe-null 0.1708
nh = np.arange(401, 501)
yre2 = qmnt.Nfl(qmnt.smooth_zeros(nh), [2], mmax=3)
assert abs(qmnt.rms(yre2) - 0.1708) < 0.01
sh = qmnt.smooth_zeros(np.array([401, 500]))
assert 13 < (sh[1] - sh[0])/(2*math.pi/math.log(2)) < 17
assert abs(2*math.pi/math.log(2) - 9.07) < 0.01
gmid = qmnt.mean_gap(np.array([450]))[0]
assert 6.0 < (2*math.pi/math.log(2))/gmid < 7.5

# ---------- neighbor-average correlation example ----------
assert abs(-0.3/math.sqrt((1 - 0.1)/2) - (-0.447)) < 1e-3

# ---------- Bessel: n = 2 die throws (text and Problem 30) ----------
Qs = [((a-(a+b)/2)**2 + (b-(a+b)/2)**2) for a, b in itertools.product(range(1, 7), repeat=2)]
assert abs(np.mean(Qs)/1 - 35/12) < 1e-12
assert abs(np.mean(Qs)/2 - 35/24) < 1e-12 and abs(35/24 - 1.458) < 1e-3

# ---------- Chebyshev and LLN numbers ----------
assert abs(1/2**2 - 0.25) < 1e-12
assert abs((1 - special.erf(2/math.sqrt(2))) - 0.0455) < 1e-4
assert abs(1/(1000*0.1**2) - 0.1) < 1e-12

# ---------- Sample Problem: confidence interval for g ----------
se = 0.15/math.sqrt(25)
assert abs(se - 0.03) < 1e-12
assert abs(1.96*se - 0.0588) < 1e-12
assert abs(9.79 - 1.96*se - 9.731) < 1e-3 and abs(9.79 + 1.96*se - 9.849) < 1e-3
assert abs(stats.t.ppf(0.975, 24) - 2.06) < 0.01

# ---------- random phase expectations ----------
Ec = integrate.quad(lambda th: math.cos(th)/(2*math.pi), 0, 2*math.pi)[0]
Ec2 = integrate.quad(lambda th: math.cos(th)**2/(2*math.pi), 0, 2*math.pi)[0]
assert abs(Ec) < 1e-12 and abs(Ec2 - 0.5) < 1e-12

# ---------- Paper connection: null and oracle values via qmnt ----------
z = qmnt.zeros(1000)
n1 = np.arange(401, 501)
null1 = qmnt.rms(qmnt.fluct(z[n1-1], n1))
assert abs(null1 - 0.2631) < 0.0015
n2 = np.arange(501, 1001)
null2 = qmnt.rms(qmnt.fluct(z[n2-1], n2))
assert abs(null2 - 0.270) < 0.0015            # Problem 25
r6, c6 = qmnt.score(qmnt.oracle_zeros(n1, qmnt.first_primes(6)), n1)
assert abs(r6 - 0.074) < 0.002 and abs(c6 - 0.96) < 0.01
r2, c2 = qmnt.score(qmnt.oracle_zeros(n1, [2]), n1)   # Problem 26
assert abs(r2 - 0.198) < 0.002 and abs(c2 - 0.66) < 0.01
assert abs(0.263*math.sqrt(1 - c2**2) - 0.1976) < 0.01
# paper interval [-0.21, +0.08] has half-width about 0.15
assert abs((0.08 - (-0.21))/2 - 0.145) < 1e-12

# ---------- Problems ----------
assert abs(3/6 - 0.5) < 1e-12   # P1
L2 = np.array([2, 4, 4, 4, 5, 5, 7, 9.])   # P2
assert L2.mean() == 5 and abs(L2.std() - 2) < 1e-12
assert abs(L2.std(ddof=1) - 2.138) < 1e-3
assert abs(np.mean(L2**2) - 29) < 1e-12 and abs(math.sqrt(29) - 5.385) < 1e-3
e1 = np.array([0.2, -0.2, 0.2, -0.2]); e2 = np.array([0.2]*4)   # P3
assert abs(np.sqrt(np.mean(e1**2)) - 0.2) < 1e-12 and abs(e1.std() - 0.2) < 1e-12 and abs(e1.mean()) < 1e-12
assert abs(np.sqrt(np.mean(e2**2)) - 0.2) < 1e-12 and abs(e2.std()) < 1e-12
X = np.array([1., 2, 5, 7]); assert abs(np.var(3*X + 4) - 9*np.var(X)) < 1e-12   # P4
assert abs((1 - math.exp(-0.5)) - 0.3935) < 1e-4 and abs(math.exp(-0.5) - 0.606531) < 1e-6   # P5
mean_u = integrate.quad(lambda x: x, -0.5, 0.5)[0]   # P6
ms_u = integrate.quad(lambda x: x**2, -0.5, 0.5)[0]
assert abs(mean_u) < 1e-12 and abs(math.sqrt(ms_u) - 0.2887) < 1e-4
assert abs(1 - 0.9545 - 0.0455) < 1e-12   # P7
assert abs(np.corrcoef([1, 2, 3, 4], [2, 1, 4, 3])[0, 1] - 0.6) < 1e-12   # P8
xa = np.array([1., 3, 2, 7]); ya = np.array([2., 5, 1, 4])   # P9
assert abs(np.corrcoef(-2*xa + 1, ya)[0, 1] + np.corrcoef(xa, ya)[0, 1]) < 1e-12
assert abs(1/9 - 0.111) < 1e-3 and abs((1 - special.erf(3/math.sqrt(2))) - 0.0027) < 1e-4   # P10
zsc = 0.5/0.263   # P11
assert abs(zsc - 1.901) < 1e-3
assert abs(2*stats.norm.sf(zsc) - 0.0573) < 2e-4
r12 = autocorr([1, 1, -1, -1, 1, 1, -1, -1], 2)   # P12
assert abs(r12[1] - 0.125) < 1e-12 and abs(r12[2] + 0.75) < 1e-12
assert abs(0.25/20 - 0.0125) < 1e-12 and abs(1.96*0.0125 - 0.0245) < 1e-12   # P13
assert abs(38/(1000*0.1) - 0.38) < 1e-12   # P14
assert abs(0.263*math.sqrt(1 + 0.16**2) - 0.2663) < 1e-4   # P19
assert abs(0.263*math.sqrt(1 + 0.62**2) - 0.3094) < 1e-4
yv, ev, rr = 0.1708, 0.1958, 0.07   # P20
bq = -2*rr*yv; cq = yv**2 - ev**2
sp_ = (-bq + math.sqrt(bq**2 - 4*cq))/2
assert abs(sp_ - 0.1084) < 2e-4
assert abs(sp_/yv - 0.635) < 2e-3
assert abs(np.sqrt(yv**2 + sp_**2 - 2*rr*yv*sp_) - ev) < 1e-12
assert abs(-0.2/math.sqrt((1 - 0.2)/2) - (-0.316)) < 1e-3   # P22
assert abs(-0.41*math.sqrt(0.5) - (-0.290)) < 1e-3
assert abs((1 - 0.96**2)/10 - 0.00784) < 1e-12   # P23
assert abs(1.96*0.00784 - 0.0154) < 1e-4
assert abs(0.96 - 0.0154 - 0.945) < 1e-3 and abs(0.96 + 0.0154 - 0.975) < 1e-3
assert abs(1.96*(1 - 0.07**2)/10 - 0.195) < 1e-3   # about +-0.2 near r=0
xe = np.random.default_rng(0).exponential(1.0, 10000)   # P24
assert abs(xe.mean() - 1) < 0.05 and abs(xe.var() - 1) < 0.1
om = math.log(2); P = 2*math.pi/om   # P27
tt = np.linspace(0, 1000*P, 2_000_001)[:-1]
re2 = -(1/math.pi)*sum(np.sin(m*tt*om)/(m*2**(m/2)) for m in (1, 2, 3))
assert abs(np.sqrt(np.mean(re2**2)) - 0.1709) < 2e-4
cv = integrate.quad(lambda th: math.cos(th)*math.sin(th)/(2*math.pi), 0, 2*math.pi)[0]   # P28
assert abs(cv) < 1e-12
# P29 numerical instance of |r_k| <= 1
xr = np.random.default_rng(4).normal(size=50)
assert np.all(np.abs(autocorr(xr, 20)) <= 1 + 1e-12)

print('all checks passed')
