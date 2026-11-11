import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq
from scipy.integrate import quad
import qmnt

TWO_PI = 2*np.pi
def Nsm(t):
    u = np.asarray(t, float)/TWO_PI
    return u*np.log(u) - u + 7/8
def dNsm(t): return np.log(np.asarray(t, float)/TWO_PI)/TWO_PI
def smooth_zero(n):
    n = np.asarray(n, float); z = (n - 11/8)/np.e
    return TWO_PI*(n - 11/8)/np.real(lambertw(z, 0))
def gap(t): return TWO_PI/np.log(np.asarray(t, float)/TWO_PI)
def rms(a): return np.sqrt(np.mean(np.asarray(a, float)**2))
def corr(a, b): return np.corrcoef(a, b)[0, 1]

tz = np.asarray(qmnt.zeros(10000), float)
t1 = tz[0]

# Data anchors quoted from the paper (Sec. II A, App. A)
assert abs(t1 - 14.134725141734695) < 1e-9
assert abs(tz[499] - 811.1843588465) < 1e-8
assert abs(tz[999] - 1419.42248094599) < 1e-8

# Sample Problem: unfolding the first zero
s1 = float(smooth_zero(1))
assert abs(Nsm(s1) - 0.5) < 1e-10
assert abs(s1 - 14.521) < 1e-3
assert abs(float(np.asarray(qmnt.smooth_zeros(np.array([1])))[0]) - s1) < 1e-6
g1 = float(gap(s1)); assert abs(g1 - 7.500) < 1e-3
f1 = (t1 - s1)/g1; assert abs(f1 - (-0.0515)) < 5e-4
assert abs(Nsm(14.0) - 0.432) < 1e-3 and abs(Nsm(15.0) - 0.565) < 1e-3   # bracket
# Newton iterates (Step 3)
assert abs(dNsm(14.0) - 0.1275) < 5e-5
assert abs(Nsm(14.0) - 0.4320) < 5e-5
x1 = 14.0 + (0.5 - Nsm(14.0))/dNsm(14.0)
assert abs(x1 - 14.533) < 5e-4
assert abs(Nsm(x1) - 0.5016) < 1e-4 and abs(dNsm(x1) - 0.1335) < 1e-4
step = (Nsm(x1) - 0.5)/dNsm(x1); assert abs(step - 0.0120) < 2e-4
x2 = x1 - step; assert abs(x2 - 14.521) < 5e-4
x3 = x2 + (0.5 - Nsm(x2))/dNsm(x2); assert abs(x3 - x2) < 1e-5

# Sample Problem: Lambert W
z1 = (1 - 11/8)/np.e
assert abs(z1 - (-0.13795)) < 1e-5
w1 = np.real(lambertw(z1, 0)); assert abs(w1 - (-0.16225)) < 1e-5
assert abs(Nsm(smooth_zero(500)) - 499.5) < 1e-9

# Sample Problem: units
assert abs(gap(811.1843588465) - 1.293) < 5e-4
assert abs(0.263*gap(811.1843588465) - 0.340) < 1e-3
assert abs(gap(1419.42248094599) - 1.159) < 5e-4

# Nulls (text, pycode): smooth-zero nulls vs paper anchors
n_all = np.arange(1, 10001)
s_all = smooth_zero(n_all); g_all = gap(s_all)
f_all = (tz - s_all)/g_all
def null(lo, hi): return rms(f_all[lo-1:hi])
assert abs(null(1, 400) - 0.2438) < 0.004
assert abs(null(401, 500) - 0.2631) < 0.002
assert abs(null(801, 1000) - 0.2739) < 0.003
nq = np.arange(401, 501)
assert abs(qmnt.rms(qmnt.fluct(tz[nq-1], nq)) - 0.2631) < 0.002

# y_n + f_n relation (eq. ynfn, Problem 15)
y_all = n_all - 0.5 - Nsm(tz)
d500 = tz[499] - s_all[499]
assert abs((y_all[499] + f_all[499]) - (-6.8e-6)) < 3e-7
assert abs((y_all[0] + f_all[0]) - (-8.3e-4)) < 1e-4
assert np.all(y_all + f_all <= 1e-12)

# np.gradient endpoint pitfall (window 1--400)
s2 = float(smooth_zero(2)); assert abs((s2 - s1) - 6.13) < 0.01
s400 = s_all[:400]; ggrad = np.gradient(s400)
assert abs(rms((tz[:400]-s400)/ggrad) - rms(f_all[:400])) < 1e-4

# GUE check: mean unfolded spacing and telescoping identity
sp = np.diff(Nsm(tz)); assert abs(sp.mean() - 1) < 1e-3
assert abs(sp.mean() - (1 + (y_all[0] - y_all[-1])/9999)) < 1e-9

# Sample Problem: L from Eq. (2)
V0 = 3.10073*np.pi
assert abs(V0 - 9.7412) < 1e-4
lt = np.log(V0/(TWO_PI*np.e**2)); assert abs(lt - (-1.5615)) < 1e-4
assert abs(np.sqrt(1000 - V0) - 31.4684) < 1e-4
assert abs(np.sqrt(V0) - 3.1211) < 1e-4
def x_of_V(V):
    a = np.sqrt(V - V0); b = np.sqrt(V)
    return (a*lt + b*2*np.log((b + a)/np.sqrt(V0)))/np.pi
def x_naive(V):
    a = np.sqrt(V - V0); b = np.sqrt(V)
    return (a*lt + b*np.log((b + a)/(b - a)))/np.pi
assert abs(x_of_V(1000.0) - x_naive(1000.0)) < 1e-8
assert abs(np.pi*x_of_V(1000.0) - 141.003) < 1e-3
assert abs(x_of_V(1000.0) - 44.883) < 5e-4
assert abs(x_of_V(1500.0) - 59.957) < 5e-4
assert abs(qmnt.L_of(1000) - 44.883) < 1e-3
assert abs(qmnt.L_of(1500) - 59.957) < 1e-3
assert abs(x_of_V(V0)) < 1e-12
Vg = np.linspace(V0 + 1e-9, 1600, 5000); assert np.all(np.diff(x_of_V(Vg)) > 0)

# grid spacings
L1 = 44.883
assert abs(2*L1/6001 - 0.0150) < 1e-4 and abs(2*59.957/8001 - 0.0150) < 1e-4

# Sample Problem: dispersion in the box
h = 2*L1/6001; assert abs(h - 0.0149585) < 1e-7
k = 756
Ec = (k*np.pi/(2*L1))**2; assert abs(Ec - 700.03) < 0.01
th = k*np.pi/(2*6001); assert abs(th - 0.197887) < 1e-6
Ed = 4/h**2*np.sin(th)**2; assert abs(Ed - 690.94) < 0.01
assert abs((Ec - Ed) - 9.09) < 0.01
assert abs(gap(700.0) - 1.333) < 5e-4
assert abs((Ec - Ed)/gap(700.0) - 6.8) < 0.05
assert abs(2*L1*np.sqrt(700)/np.pi - 755.98) < 0.01

# Sample Problem: Richardson subscript
E, a, hh = 100.0, -50.0, 0.02
Eh, Eh2, E2h = E + a*hh**2, E + a*(hh/2)**2, E + a*(2*hh)**2
assert abs((4*Eh2 - Eh)/3 - 100.000) < 1e-9
assert abs((4*E2h - Eh)/3 - 99.900) < 1e-9
assert abs(qmnt.richardson(Eh, Eh2) - 100.0) < 1e-9

# Sample Problem: rms bound
f = np.array([0.30, -0.20, 0.10, -0.40]); e = np.array([0.010, -0.020, 0.005, 0.015])
assert abs(rms(f) - 0.2739) < 1e-4 and abs(rms(f - e) - 0.2728) < 1e-4
assert abs(abs(rms(f - e) - rms(f)) - 0.0010) < 1e-4
assert abs(rms(e) - 0.0137) < 1e-4
assert abs(rms(f-e)-rms(f)) <= rms(e) <= np.max(np.abs(e))

# Sample Problem: leakage numbers
t_n, Eg, ER, Ef = 700.250, 698.300, 700.100, 698.600
D = Eg - ER; assert abs(D - (-1.800)) < 1e-9
tt = t_n + D; assert abs(tt - 698.450) < 1e-9
assert abs((tt - Ef) - (-0.150)) < 1e-9 and abs((t_n - (Ef - D)) - (-0.150)) < 1e-9
assert abs((t_n - ER) - 0.150) < 1e-9

# Sample Problem: Table V entries via qmnt
def oracle_score(nn, P):
    tp = np.asarray(qmnt.oracle_zeros(nn, P, mmax=3), float)
    return qmnt.score(tp, nn)
r, c = oracle_score(nq, [2]); assert abs(r - 0.198) < 0.003 and abs(c - 0.66) < 0.01
r, c = oracle_score(nq, list(qmnt.first_primes(6))); assert abs(r - 0.074) < 0.003 and abs(c - 0.96) < 0.01

# Why 500 levels: N_sm(1000), count, turning-point margins
assert abs(Nsm(1000.0) - 648.6) < 0.05
assert int(np.sum(s_all <= 1000)) == 649
assert tz[499] < 1000 and tz[999] < 1500
xt1 = x_of_V(811.18); xt2 = x_of_V(1419.42)
assert abs(xt1 - 38.53) < 0.02 and abs(xt2 - 57.66) < 0.02
assert abs((L1 - xt1) - 6.35) < 0.02 and abs((59.957 - xt2) - 2.29) < 0.02

# Problem 2, 3
assert abs(Nsm(100.0) - 29.00) < 0.01
assert int(np.sum(tz <= 100)) == 29
assert abs(gap(100.0) - 2.271) < 5e-4

# Problem 5
xa = np.sqrt(1000)/np.pi*np.log(2000/(np.pi*np.e**2))
assert abs(xa - 44.86) < 0.005 and abs(np.log(2000/(np.pi*np.e**2)) - 4.4562) < 1e-4

# Problem 6
s500 = float(smooth_zero(500)); assert abs(s500 - 811.448) < 0.002
assert abs((500 - 11/8)/np.e - 183.43) < 0.01
assert abs(float(np.asarray(qmnt.smooth_zeros(np.array([500])))[0]) - s500) < 1e-5
assert abs(gap(s500) - 1.2926) < 1e-4
assert abs((tz[499] - s500)/gap(s500) - (-0.204)) < 0.002
assert abs(d500 - (-0.2635)) < 5e-4

# Problem 8: Richardson in the empty box
def Edisc(kk, N): hh_ = 2*L1/(N + 1); return 4/hh_**2*np.sin(kk*np.pi/(2*(N + 1)))**2
Ek = (100*np.pi/(2*L1))**2; assert abs(Ek - 12.2483) < 1e-4
e1 = Edisc(100, 1000) - Ek; assert abs(e1 - (-0.100)) < 1e-3
eR = (4*Edisc(100, 2001) - Edisc(100, 1000))/3 - Ek
assert -9e-5 < eR < -7.5e-5
assert abs(Edisc(100, 1000) - 12.1481) < 1e-4
Efd = np.sort(np.asarray(qmnt.fd_levels(np.zeros(1000), L1, k=100), float))
assert abs(Efd[99] - Edisc(100, 1000)) < 1e-6

# Problem 10
assert abs(null(501, 1000) - 0.270) < 0.003
assert abs(null(1001, 2000) - 0.279) < 0.003

# Problem 11
n501 = np.arange(501, 1001)
r, c = oracle_score(n501, [2]); assert abs(r - 0.208) < 0.003 and abs(c - 0.64) < 0.01
r, c = oracle_score(n501, [2, 3, 5]); assert abs(r - 0.127) < 0.003 and abs(c - 0.88) < 0.01

# Problem 12: within-height random split null
rng = np.random.default_rng(1)
test = rng.choice(400, 80, replace=False) + 1
assert 400 - len(test) == 320
nr = rms(f_all[test - 1]); assert 0.20 < nr < 0.29

# Problem 13: odd/even nulls
assert abs(rms(f_all[1:400:2]) - 0.238) < 0.006
assert abs(rms(f_all[0:400:2]) - 0.250) < 0.006

# Problem 19
p = np.array([0.5, -0.5, 0.0]); q = np.array([-1.0, 1.0, 0.0]); d = np.array([0, 0, 3.0])
assert abs(corr(p, q) - (-1)) < 1e-12
assert abs(corr(p + d, q + d) - 5/np.sqrt(52)) < 1e-12 and abs(5/np.sqrt(52) - 0.693) < 1e-3
assert abs(rms(p - q) - 1.225) < 1e-3 and abs(rms((p+d) - (q+d)) - rms(p - q)) < 1e-12

# Problem 20: GUE normalization and mean
pg = lambda s: 32/np.pi**2*s**2*np.exp(-4*s**2/np.pi)
assert abs(quad(pg, 0, np.inf)[0] - 1) < 1e-10
assert abs(quad(lambda s: s*pg(s), 0, np.inf)[0] - 1) < 1e-10

# Problem 21
Pg = quad(pg, 0, 0.2)[0]; assert abs(Pg - 0.0084) < 3e-4
assert abs(1 - np.exp(-0.2) - 0.181) < 1e-3
frac = np.mean(sp < 0.2); assert frac < 0.05

# Problem 23: own oracle with brentq
def Nfl(t, P):
    return -sum(np.sin(m*t*np.log(pp))/(m*pp**(m/2)) for pp in P for m in (1, 2, 3))/np.pi
tp = []
for nn in nq:
    sn, gn = s_all[nn-1], g_all[nn-1]
    F = lambda t: Nsm(t) + Nfl(t, [2, 3, 5]) - (nn - 0.5)
    lo, hi = sn - 3*gn, sn + 3*gn
    while F(lo)*F(hi) > 0: lo -= gn; hi += gn
    tp.append(brentq(F, lo, hi))
tp = np.array(tp)
assert abs(rms((tp - tz[nq-1])/g_all[nq-1]) - 0.116) < 0.003
assert abs(corr((tp - s_all[nq-1])/g_all[nq-1], f_all[nq-1]) - 0.90) < 0.01

# Problem 25
t10k = tz[-1]; assert abs(t10k - 9877.78) < 0.01
assert abs(gap(t10k) - 0.854) < 2e-3
assert abs(5e-12/gap(t10k) - 6e-12) < 1e-12
assert abs(0.018/(6e-12) - 3e9) < 1e8

print('all checks passed')
