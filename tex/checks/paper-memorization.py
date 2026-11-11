import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import qmnt

PI = np.pi
V0 = 3.10073*PI

def xV(V):
    V = float(V); a = np.sqrt(max(V - V0, 0.0)); b = np.sqrt(V)
    return (a*np.log(V0/(2*PI*np.e**2)) + b*np.log((b + a)/(b - a)))/PI

def Nsm(t):
    u = t/(2*PI); return u*np.log(u) - u + 7/8

def szero(n):
    return brentq(lambda t: Nsm(t) - (n - 0.5), 7.0, 1e6)

def gap(s):
    return 2*PI/np.log(np.asarray(s)/(2*PI))

def Nfl(t, primes, mmax=3):
    t = np.atleast_1d(np.asarray(t, float)); out = np.zeros_like(t)
    for p in primes:
        for m in range(1, mmax + 1):
            out += np.sin(m*t*np.log(p))/(m*p**(m/2))
    return -out/PI

def rms(a): return np.sqrt(np.mean(np.asarray(a)**2))
def corr(a, b): return np.corrcoef(a, b)[0, 1]

t = np.asarray(qmnt.zeros(1000), float)

# --- basic data anchors (paper App. A) ---
assert abs(t[0] - 14.1347251417) < 1e-6
assert abs(t[499] - 811.1843588465) < 1e-6
assert abs(t[99] - 236.524230) < 1e-5          # Problem 11 input
assert abs(t[999] - 1419.42248) < 1e-4          # Problem 6 input

# --- Sample: turning points; constants ---
assert abs(V0 - 9.74123) < 1e-4
assert abs(np.log(V0/(2*PI*np.e**2)) - (-1.56151)) < 1e-4
assert abs(2*PI*np.e**2 - 46.4268) < 1e-3
L1 = xV(1000.0); L2 = xV(1500.0)
assert abs(L1 - 44.883) < 1e-3
assert abs(L2 - 59.957) < 1e-3
assert abs(xV(500.0) - 26.82) < 0.01
assert abs(np.log((np.sqrt(500) + np.sqrt(500 - V0))**2/V0) - 5.31473) < 1e-4
assert abs(xV(t[200]) - 22.5) < 0.15             # band edge, level 201
assert abs(xV(t[299]) - 28.5) < 0.15             # band edge, level 300
assert abs(xV(680.0) - 33.82) < 0.01             # forward-split reading
assert abs(xV(811.1843588) - 38.53) < 0.01
assert abs(xV(t[399]) - 33.8) < 0.15
assert abs(xV(t[99]) - 14.81) < 0.01             # Problem 11

# --- Sample: coverage law at scale 1 ---
Mneed = 2*L1*np.sqrt(811.1843588)/PI
assert abs(np.sqrt(811.1843588) - 28.4813) < 1e-4
assert abs(Mneed - 813.8) < 0.1
assert abs((900*PI/(2*L1))**2 - 992.1) < 0.2
assert abs((300*PI/(2*L1))**2 - 110.2) < 0.1
assert np.sum(t < (300*PI/(2*L1))**2) == 33
assert abs(t[32] - 107.17) < 0.01 and abs(t[33] - 111.03) < 0.01
assert abs(2*L1*np.sqrt(680.0)/PI - 745) < 2
s400 = szero(400)
assert abs(s400 - 679.87) < 0.1
assert abs(2*L1*np.sqrt(t[399])/PI - 745) < 3
# Problem 6: scale 2
assert abs(2*L2*np.sqrt(1419.42248)/PI - 1438) < 0.5

# --- Sample: relative skill ---
assert abs((1 - 0.369/0.263) - (-0.403)) < 1e-3
assert abs((1 - 0.265/0.263) - (-0.0076)) < 1e-4
assert abs((1 - 0.313/0.263) - (-0.190)) < 1e-3
assert abs((1 - 0.074/0.263) - 0.719) < 1e-3
assert abs((1 - 0.198/0.263) - 0.247) < 1e-3
assert abs(0.265/0.198 - 1.34) < 0.005

# --- Sample: uncorrelated prediction ---
assert abs(0.263*np.sqrt(1 + 0.16**2) - 0.2663) < 1e-4
assert abs(0.263*np.sqrt(1 + 0.62**2) - 0.3094) < 1e-4
# numerical instance of Eq. decomp
rng = np.random.default_rng(0)
pp = rng.normal(0.1, 0.3, 1000); dd = 0.4*pp + rng.normal(-0.05, 0.25, 1000)
lhs = np.mean((pp - dd)**2)
rhs = (pp.mean() - dd.mean())**2 + pp.var() + dd.var() - 2*corr(pp, dd)*pp.std()*dd.std()
assert abs(lhs - rhs) < 1e-12

# --- Sample: halves ---
assert abs(np.sqrt((0.321**2 + 0.304**2)/2) - 0.3126) < 1e-4

# --- Sample: neighbor anticorrelation ---
s2, _ = quad(lambda u: u**2*32/PI**2*u**2*np.exp(-4*u**2/PI), 0, np.inf)
assert abs(s2 - 3*PI/8) < 1e-9 and abs(s2 - 1.1781) < 1e-4
rho1 = 1 - (s2 - 1)/(2*0.263**2)
assert abs(rho1 - (-0.287)) < 1e-3
assert abs(-0.29/np.sqrt(0.45) - (-0.432)) < 1e-3
# numerical instance of Eq. nbrcorr with an AR-like correlated series
z = rng.normal(size=200000)
seq = z - 0.4*np.roll(z, 1)                      # MA(1): negative lag-1 corr
def ac(a, lag):
    a = a - a.mean(); return np.sum(a[:-lag]*a[lag:])/np.sum(a*a)
r1, r2 = ac(seq, 1), ac(seq, 2)
qn = 0.5*(seq[:-2] + seq[2:]); dn = seq[1:-1]
assert abs(corr(qn, dn) - r1/np.sqrt((1 + r2)/2)) < 0.01

# --- nulls and oracle (paper Sec. II C, Table V) ---
n_out = np.arange(401, 501)
s_out = np.array([szero(k) for k in n_out]); g_out = gap(s_out)
d_out = (t[n_out - 1] - s_out)/g_out
assert abs(rms(d_out) - 0.2631) < 0.003
n_in = np.arange(1, 401)
s_in = np.array([szero(k) for k in n_in]); g_in = gap(s_in)
d_in = (t[n_in - 1] - s_in)/g_in
assert abs(rms(d_in) - 0.2438) < 0.01

def oracle(n, primes):
    s = szero(n); g = gap(s)
    f = lambda x: Nsm(x) + Nfl(x, primes)[0] - (n - 0.5)
    lo, hi = s - 3*g, s + 3*g
    while f(lo)*f(hi) > 0:
        lo -= g; hi += g
    return brentq(f, lo, hi)
P6 = [2, 3, 5, 7, 11, 13]
tp = np.array([oracle(k, P6) for k in n_out]); pf = (tp - s_out)/g_out
assert abs(rms(pf - d_out) - 0.074) < 0.003      # Problem 17, Table V
assert abs(corr(pf, d_out) - 0.96) < 0.01
tp2 = np.array([oracle(k, [2]) for k in n_out]); pf2 = (tp2 - s_out)/g_out
assert abs(rms(pf2 - d_out) - 0.198) < 0.004     # Problem 17, {2} column
assert abs(corr(pf2, d_out) - 0.66) < 0.02

# --- autocorrelation signs (paper Sec. III; Problem 15) ---
assert ac(d_in, 1) < -0.1
assert ac(d_in, 2) < 0
assert ac(d_in, 3) < 0

# --- Sample: surrogates ---
assert abs(abs(0.137 - 0.134)/0.010 - 0.30) < 1e-9
assert abs(abs(0.137 - 0.141)/0.006 - 0.667) < 1e-3
assert abs(abs(0.137 - 0.138)/0.005 - 0.20) < 1e-9
assert abs((0.244 - 0.137)/0.244 - 0.439) < 1e-3
assert abs((0.244 - 0.134)/0.244 - 0.451) < 1e-3
# phase randomization preserves rms (listing)
r = np.random.default_rng(1)
D = np.fft.rfft(d_in); ph = r.uniform(0, 2*PI, len(D)); ph[0] = 0.0; ph[-1] = 0.0
dsur = np.fft.irfft(D*np.exp(1j*ph), n=len(d_in))
assert abs(rms(dsur) - rms(d_in)) < 1e-12
assert abs(rms(r.permutation(d_in)) - rms(d_in)) < 1e-12

# --- interpolation split arithmetic ---
assert abs(2.64/0.255 - 10.4) < 0.05
assert abs(40/15 - 2.7) < 0.05 and abs(40/13 - 3.1) < 0.05
assert abs(199/500 - 0.40) < 0.005

# --- Sample: RE2 rms and probe-null ---
a = np.array([1/(m*2**(m/2)) for m in (1, 2, 3)])
assert abs(a[0] - 0.707107) < 1e-6 and abs(a[2] - 0.117851) < 1e-6
assert abs(np.sum(a**2) - 0.5763889) < 1e-7
re2rms = np.sqrt(0.5*np.sum(a**2))/PI
assert abs(re2rms - 0.17088) < 1e-5
assert abs(rms(Nfl(s_out, [2])) - 0.1708) < 0.002
assert abs(0.0555/0.1709 - 0.32) < 0.01
assert abs(0.1958/0.1708 - 1.146) < 1e-3          # Problem 25

# --- Sample: levels per period ---
per = 2*PI/np.log(2)
assert abs(per - 9.0647) < 1e-4
assert abs(np.log(680/(2*PI))/np.log(2) - 6.758) < 1e-3
assert abs(np.log(811.18/(2*PI))/np.log(2) - 7.012) < 1e-3
assert abs((680 - 14.13)/per - 73.4) < 0.05
assert abs((811.18 - 680)/per - 14.5) < 0.05
assert abs((t[399] - t[0])/per - 73.4) < 0.3
assert abs((t[499] - t[399])/per - 14.5) < 0.2

# --- Problems ---
assert abs(0.313/0.064 - 4.89) < 0.005            # P1
assert abs(0.313/0.263 - 1.19) < 0.005
aa = np.sqrt((0.313/0.263)**2 - 1)                 # P5a
assert abs(aa - 0.645) < 1e-3
ab = 0.5*(-0.32 + np.sqrt(0.32**2 + 4*((0.313/0.263)**2 - 1)))   # P5b
assert abs(ab - 0.505) < 1e-3
assert abs(1/(2*PI) - 0.159) < 5e-4                # P7
assert abs(0.5/np.sum(a**2) - 0.867) < 1e-3
assert abs(np.log(680/(2*PI))/np.log(3) - 4.264) < 1e-3     # P8
assert abs(np.log(680/(2*PI))/np.log(5) - 2.910) < 1e-3
assert abs(np.sqrt((400*0.136**2 + 100*2.64**2)/500) - 1.187) < 1e-3   # P9
assert abs(0.312/0.238 - 1.311) < 1e-3             # P10
assert abs(0.311/0.250 - 1.244) < 1e-3
assert abs(0.062/0.238 - 0.2605) < 1e-3
assert abs((1 - 0.117/0.245) - 0.52) < 0.005       # P12
assert abs(0.57/0.245 - 2.33) < 0.005 and abs(0.70/0.245 - 2.86) < 0.005
assert abs(-0.29/0.90 - (-0.322)) < 1e-3           # P13
assert abs(0.29/np.sqrt(0.45) - 0.432) < 1e-3
assert abs((0.2438 - 0.206)/(0.2438 - 0.137) - 0.354) < 1e-3   # P19
assert abs((0.265 - 0.263) - 0.002) < 1e-9
assert abs(0.296/0.263 - 1.125) < 1e-3             # P20
dphi = (PI/(680 - 14.13))*(811.18 - 680)           # P21
assert abs(dphi - 0.62) < 0.005
dphi_true = (PI/(t[399] - t[0]))*(t[499] - t[399])
assert abs(dphi_true - 0.62) < 0.02
assert abs(0.010/np.sqrt(10) - 0.0032) < 1e-4      # P23
assert abs(0.003/(0.010/np.sqrt(10)) - 0.95) < 0.005

print("all checks passed")
