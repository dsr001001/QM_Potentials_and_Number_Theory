import numpy as np, mpmath, qmnt
from scipy.optimize import brentq
from scipy.special import lambertw
from scipy.integrate import quad
from scipy.signal import find_peaks

def close(x, y, tol, msg):
    assert abs(x - y) < tol, (msg, x, y)

def Nsm(t):
    t = np.asarray(t, float)
    return t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8

def dNsm(t):
    return np.log(t/(2*np.pi))/(2*np.pi)

def Nfl(t, P, mmax=3):
    t = np.asarray(t, float); s = np.zeros_like(t)
    for p in P:
        for m in range(1, mmax+1):
            s = s + np.sin(m*t*np.log(p))/(m*p**(m/2))
    return -s/np.pi

def dNfl(t, P, mmax=3):
    s = 0.0
    for p in P:
        for m in range(1, mmax+1):
            s = s + np.log(p)*np.cos(m*t*np.log(p))/p**(m/2)
    return -s/np.pi

def smooth_zero(n):
    return brentq(lambda t: float(Nsm(t)) - (n - 0.5), 5.0, 1e5)

t1 = 14.134725141734695
close(qmnt.zeros(1)[0], t1, 1e-9, 't1')

# Sample: log zeta(2) from primes 2,3,5,7
lz2 = np.log(np.pi**2/6); close(lz2, 0.4977003, 1e-6, 'log zeta(2)')
terms = [np.log(p*p/(p*p-1.0)) for p in [2, 3, 5, 7]]
for v, c in zip(terms, [0.2876821, 0.1177830, 0.0408220, 0.0206193]):
    close(v, c, 1e-6, 'SP1 terms')
close(sum(terms), 0.4669064, 1e-6, 'SP1 sum')
close(sum(terms)/lz2, 0.938, 0.0005, 'SP1 fraction')
close(lz2 - sum(terms), 0.0307943, 1e-6, 'SP1 remainder')
pp = np.asarray(qmnt.primes_upto(10**6), float)
close(np.sum(-np.log(1 - pp**-2.0)), lz2, 1e-5, 'Euler log series')

# Sample: 3^{-s} at s=1/2+10i
z3 = 3.0**(-(0.5 + 10j))
close(z3.real, -0.005457, 2e-6, 'SP2 real'); close(z3.imag, 0.577324, 2e-6, 'SP2 imag')
close(-np.sin(10*np.log(3))/np.sqrt(3)/np.pi, 0.18377, 2e-5, 'SP2 wave')

# Amplitude table
tab = [(2, 1, 0.6931, 9.0647, 0.7071), (3, 1, 1.0986, 5.7192, 0.5774), (2, 2, 1.3863, 4.5324, 0.2500),
       (5, 1, 1.6094, 3.9040, 0.4472), (7, 1, 1.9459, 3.2289, 0.3780), (2, 3, 2.0794, 3.0216, 0.1179),
       (3, 2, 2.1972, 2.8596, 0.1667), (11, 1, 2.3979, 2.6203, 0.3015), (13, 1, 2.5649, 2.4496, 0.2774)]
for p, m, w, per, amp in tab:
    k = p**m
    close(np.log(k), w, 6e-5, 'tab omega'); close(2*np.pi/np.log(k), per, 6e-5, 'tab period')
    close(1/(m*np.sqrt(k)), amp, 6e-5, 'tab amp')
close(np.log(25), 3.2189, 6e-5, 'checkpoint 25')

# RE2 periods and rms
close(2*np.pi/np.log(2), 9.0647, 1e-4, 'RE2 period')
close(2*np.pi/(2*np.log(2)), 4.5324, 1e-4, 'RE2 period 2')
close(2*np.pi/(3*np.log(2)), 3.0216, 1e-4, 'RE2 period 3')
ms = sum(1/(m*m*2**m) for m in (1, 2, 3)); close(ms, 0.5763889, 1e-7, 'RE2 sum')
re2rms = np.sqrt(ms/2)/np.pi; close(re2rms, 0.1709, 1e-4, 'RE2 rms')
tt = np.linspace(0, 2*np.pi/np.log(2), 200001)
close(np.sqrt(np.mean(Nfl(tt, [2])**2)), re2rms, 1e-4, 'RE2 rms numeric')
sn = np.array([smooth_zero(n) for n in range(401, 501)])
close(np.sqrt(np.mean(Nfl(sn, [2])**2)), 0.1708, 0.003, 'probe null')
gaps = 2*np.pi/np.log(sn/(2*np.pi))
assert 1.25 < gaps.min() and gaps.max() < 1.45, ('gaps', gaps.min(), gaps.max())
assert gaps.max() < 3.0216/2, 'Nyquist'
nper = (sn[-1] - sn[0])/(2*np.pi/np.log(2))
assert 13.8 < nper < 15.0, ('periods spanned', nper)

# Sum of 1/sqrt(p) for p<=100
p100 = np.asarray(qmnt.primes_upto(100), float)
assert len(p100) == 25
close(np.sum(p100**-0.5), 5.5365, 2e-4, 'sum p^-1/2')
close(np.sum(p100**-0.5)/np.pi, 1.76, 0.005, 'max m=1 amplitude')
ratio = np.sum(pp**-0.5)/(2*np.sqrt(1e6)/np.log(1e6))
assert 0.9 < ratio < 1.4, ('2sqrt x/log x', ratio)

# Sample: N_fl at t1, N_sm(t1), S(t1)
close(float(Nfl(t1, [2])), 0.0617, 3e-4, 'SP Nfl(t1)')
close(float(Nsm(t1)), 0.44928, 2e-5, 'SP Nsm(t1)')
close(0.5 - float(Nsm(t1)), 0.05072, 2e-5, 'SP S(t1)')
close(1/(48*np.pi*t1), 0.0005, 5e-5, '1/t correction')
close(t1/(2*np.pi), 2.2496113, 1e-6, 't1/2pi')
L2 = np.log(2)
ph = [m*t1*L2 for m in (1, 2, 3)]
for v, c in zip(ph, [9.797445, 19.594890, 29.392335]):
    close(v, c, 2e-6, 't1 phases')
for m, (v, c) in enumerate(zip(ph, [0.372667, 0.745334, 1.118001]), 1):
    close(v - 3*m*np.pi, c, 2e-6, ('t1 residue', m))
sins = [np.sin(v) for v in ph]
for v, c in zip(sins, [-0.364101, 0.678217, -0.899228]):
    close(v, c, 3e-6, 't1 sines')
wm = [m*2**(m/2) for m in (1, 2, 3)]
for v, c in zip(wm, [1.414214, 4.0, 8.485281]):
    close(v, c, 1e-6, 'weights m 2^(m/2)')
trm = [s/w for s, w in zip(sins, wm)]
for v, c in zip(trm, [-0.257458, 0.169554, -0.105975]):
    close(v, c, 3e-6, 't1 terms')
close(sum(trm), -0.193879, 3e-6, 't1 term sum')
close(-sum(trm)/np.pi, 0.06171, 1e-5, 'Nfl(t1) from terms')
close(1/np.sqrt(97), 0.101535, 1e-6, '1/sqrt(97)')

# Sample: predicting t1 with p=2
s1 = smooth_zero(1); close(s1, 14.5214, 2e-4, 's1')
g1 = 2*np.pi/np.log(s1/(2*np.pi)); close(g1, 7.500, 2e-3, 'gap1')
close(np.log(s1/(2*np.pi)), 0.837746, 2e-5, 'log s1/2pi')
close(np.log(t1/(2*np.pi)), 0.8107574, 1e-6, 'log t1/2pi')
close(dNsm(t1), 0.12904, 2e-5, 'slope t1')
step1 = (0.5 - float(Nsm(t1)))/dNsm(t1); close(step1, 0.3931, 2e-4, 'Newton step 1')
it1 = t1 + step1; close(it1, 14.5278, 3e-4, 'Newton iterate 1')
close(float(Nsm(it1)) - 0.5, 8.5e-4, 1e-4, 'Nsm(it1)-1/2')
close(dNsm(it1), 0.1334, 2e-4, 'slope it1')
step2 = -(float(Nsm(it1)) - 0.5)/dNsm(it1); close(step2, -0.0064, 5e-4, 'Newton step 2')
close(it1 + step2, 14.5214, 2e-4, 'Newton iterate 2')
assert abs(it1 + step2 - s1) < 1e-5, 'Newton converged'
nf = float(Nfl(s1, [2])); close(nf, 0.09350, 2e-4, 'Nfl(s1)')
dnf = dNfl(s1, [2]); close(dnf, 0.0667, 3e-4, 'Nfl prime(s1)')
ps = [m*s1*L2 for m in (1, 2, 3)]
close(ps[0], 10.0654, 3e-4, 's1 log2')
for m, c in zip((1, 2, 3), [0.6407, 1.2813, 1.9220]):
    close(ps[m-1] - 3*m*np.pi, c, 3e-4, ('s1 residue', m))
for m, c in zip((1, 2, 3), [-0.5977, 0.9584, -0.9390]):
    close(np.sin(ps[m-1]), c, 3e-4, ('s1 sine', m))
for m, c in zip((1, 2, 3), [-0.8017, 0.2855, 0.3440]):
    close(np.cos(ps[m-1]), c, 3e-4, ('s1 cos', m))
st = [np.sin(ps[m-1])/(m*2**(m/2)) for m in (1, 2, 3)]
for v, c in zip(st, [-0.4227, 0.2396, -0.1107]):
    close(v, c, 3e-4, 's1 sine terms')
close(sum(st), -0.2937, 3e-4, 's1 sine sum')
ct = [np.cos(ps[m-1])/2**(m/2) for m in (1, 2, 3)]
for v, c in zip(ct, [-0.5669, 0.1427, 0.1216]):
    close(v, c, 3e-4, 's1 cos terms')
close(sum(ct), -0.3025, 3e-4, 's1 cos sum')
for v, c in zip([2**0.5, 2.0, 2**1.5], [1.414214, 2.0, 2.828427]):
    close(v, c, 1e-6, '2^(m/2)')
close(L2/np.pi, 0.22064, 1e-5, 'log2/pi')
close(g1*nf, 0.7012, 5e-4, 'gap1*Nfl(s1)')
close(s1 - g1*nf, 13.820, 3e-3, 'crude first order')
close(s1 - nf/(1/g1 + dnf), 14.054, 3e-3, 'better first order')
f = lambda t: float(Nsm(t) + Nfl(t, [2])) - 0.5
assert f(13.8202) < 0 < f(14.1347)
close(f(13.8202), -0.0635, 3e-4, 'f(13.82)'); close(f(14.1347), 0.0110, 3e-4, 'f(14.135)')
close(13.8202/(2*np.pi), 2.199553, 2e-6, '13.8202/2pi')
close(float(Nsm(13.8202)), 0.4093, 1e-4, 'Nsm(13.8202)')
for m, red, sv in [(1, 0.15465, -0.1540), (2, 0.30931, 0.3044), (3, 0.46396, -0.4475)]:
    close(m*13.8202*L2 - 3*m*np.pi, red, 1e-5, ('phase residue', m))
    close(np.sin(m*13.8202*L2), sv, 1e-4, ('sine', m))
close(float(Nfl(13.8202, [2])), 0.0272, 1e-4, 'Nfl(13.8202)')
close(float(Nsm(14.1347)), 0.4493, 1e-4, 'Nsm(14.1347)')
close(float(Nfl(14.1347, [2])), 0.0617, 2e-4, 'Nfl(14.1347)')
close(0.4493 + 0.0617 - 0.5, 0.0110, 1e-9, 'f sum a'); close(0.4093 + 0.0272 - 0.5, -0.0635, 1e-9, 'f sum b')
close(f(14.0883), 0.00036, 2e-4, 'f(14.088)')
close(14.1347 - 13.8202, 0.3145, 1e-9, 'dt1'); close(14.0883 - 13.8202, 0.2681, 1e-9, 'dt2')
close(13.8202 + 0.3145*0.0635/(0.0635 + 0.0110), 14.0883, 1e-4, 'interpolation 1')
close((0.00036 + 0.0635)/0.2681, 0.238, 5e-4, 'interpolation slope')
close(-0.00036/0.238, -0.0015, 1e-4, 'interpolation correction')
close(14.0883 - 0.0015, 14.0868, 1e-9, 'interpolated root')
root = brentq(f, 13.8, 14.2); close(root, 14.087, 2e-3, 'root t1')
close(root, 14.0868, 1.5e-3, 'root vs interpolation')
close((t1 - s1)/g1, -0.0516, 5e-4, 'true fluct 1')
close((root - s1)/g1, -0.0580, 5e-4, 'pred fluct 1')
close(abs((root - s1)/g1 - (t1 - s1)/g1), 0.0064, 5e-4, 'oracle error 1')
close(dNsm(t1), 0.129, 1e-3, 'slope at t1')

# Uniqueness threshold P={2}
B2 = np.log(2)*(1/np.sqrt(2) + 0.5 + 1/(2*np.sqrt(2)))/np.pi
close(B2, 0.34434, 1e-5, 'bound slope p=2')
close(2*np.pi*B2, 2.16353, 1e-5, 'log threshold')
close(np.exp(2*np.pi*B2), 8.7018, 1e-3, 't/2pi threshold')
close(2*np.pi*np.exp(2*np.pi*B2), 54.7, 0.05, 'threshold 54.7')
tg = np.linspace(13.82, 14.14, 3201)
assert np.all(dNsm(tg) + dNfl(tg, [2]) > 0), 'monotone on [13.82, 14.14]'
B23 = (np.log(2)*(1/np.sqrt(2)+0.5+1/(2*np.sqrt(2))) + np.log(3)*(1/np.sqrt(3)+1/3+1/(3*np.sqrt(3))))/np.pi
close(B23, 0.73010, 1e-5, 'P9 bound'); close(2*np.pi*B23, 4.58736, 1e-5, 'P9 log')
close(np.exp(2*np.pi*B23), 98.235, 0.01, 'P9 t/2pi')
close(2*np.pi*np.exp(2*np.pi*B23), 617.2, 0.1, 'P9 threshold')

# Bracket checkpoint: total amplitude of N_fl for 25 primes below 3
amp25 = sum(1/(np.pi*m*p**(m/2)) for p in qmnt.first_primes(25) for m in (1, 2, 3))
assert amp25 < 3, amp25

# Lambert W formula
for n in [1, 10, 100, 1000]:
    A = n - 11/8
    sW = (2*np.pi*A/lambertw(A/np.e, 0)).real
    close(sW, smooth_zero(n), 1e-8, 'Lambert s_n')
close(lambertw(-3/(8*np.e), 0).real, -0.16226, 1e-5, 'W(-3/8e)')
close(-3/(8*np.e), -0.13795, 1e-5, 'A/e n=1')
close((2*np.pi*(-0.375)/lambertw(-3/(8*np.e), 0)).real, 14.521, 1e-3, 's1 via W')

# Table V reproduction
z = qmnt.zeros(1000)
n1 = np.arange(401, 501); n2 = np.arange(501, 1001)
close(qmnt.rms(qmnt.fluct(z[n1-1], n1)), 0.2631, 0.001, 'null 401-500')
close(qmnt.rms(qmnt.fluct(z[n2-1], n2)), 0.270, 0.0015, 'null 501-1000')
rows = {1: [(0.198, 0.66), (0.208, 0.64)], 3: [(0.116, 0.90), (0.127, 0.88)],
        6: [(0.074, 0.96), (0.083, 0.95)], 25: [(0.043, 0.99), (0.050, 0.99)]}
for k, vals in rows.items():
    P = qmnt.first_primes(k)
    for nn, (r, c) in zip([n1, n2], vals):
        rr, cc = qmnt.score(qmnt.oracle_zeros(nn, P, mmax=3), nn)
        close(rr, r, 0.0015, ('TableV rms', k)); close(cc, c, 0.011, ('TableV corr', k))
assert list(qmnt.first_primes(6)) == [2, 3, 5, 7, 11, 13]
assert int(qmnt.first_primes(25)[-1]) == 97

# Height-dependence ratios (six primes)
close(0.074/0.263, 0.28, 0.005, 'ratio 401-500')
close(0.083/0.270, 0.31, 0.005, 'ratio 501-1000')
close(0.119/0.296, 0.40, 0.005, 'ratio 5001-10000')

# Probe slope correction (answer 23)
assert abs(gaps.max() - 1.341) < 0.0005, gaps.max()
assert 0.455 < gaps.max()*B2 < 0.465, gaps.max()*B2
op2 = qmnt.oracle_zeros(n1, qmnt.first_primes(1), mmax=3)
d2 = qmnt.fluct(op2, n1) - (-Nfl(sn, [2]))
assert qmnt.rms(d2) < 0.08, qmnt.rms(d2)

# Control B frequencies (problem 14)
assert int(qmnt.first_primes(150)[-1]) == 863
close(3*np.log(863), 20.28, 0.005, '3 log 863')
close(0.04*3*np.log(863), 0.81, 0.005, 'slope 0.81')

# Landau sample
T = z[-1]; close(T, 1419.42248, 1e-4, 't1000')
close(T/(2*np.pi), 225.908, 1e-3, 'T/2pi')
dips = {2: 110.7, 3: 143.3, 4: 78.3, 5: 162.6, 7: 166.2}
lam = {2: np.log(2), 3: np.log(3), 4: np.log(2), 5: np.log(5), 7: np.log(7)}
for k, d in dips.items():
    close(T/(2*np.pi)*lam[k]/np.sqrt(k), d, 0.06, ('Landau dip', k))
close(np.log(2)/np.sqrt(2), 0.490129, 1e-6, 'w2'); close(np.log(3)/np.sqrt(3), 0.634285, 1e-6, 'w3')
close(np.log(2)/2, 0.346574, 1e-6, 'w4')
Ssum = np.sum(np.cos(np.log(2)*z))
assert abs(Ssum + 110.7) < 0.3*110.7, ('Landau numeric', Ssum)
def vm(k):
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        q = p
        while q <= k:
            if q == k: return np.log(p)
            q *= p
    return 0.0
wts = {k: vm(k)/np.sqrt(k) for k in range(2, 21) if vm(k) > 0}
assert max(wts, key=wts.get) == 7
for k, c in [(2, 0.490), (3, 0.634), (4, 0.347), (5, 0.720), (7, 0.735), (8, 0.245), (9, 0.366),
             (11, 0.723), (13, 0.711), (16, 0.173), (17, 0.687), (19, 0.676)]:
    close(wts[k], c, 6e-4, ('P11 weight', k))
close(np.exp(2), 7.4, 0.05, 'peak of log x/sqrt x')
close(np.exp(3.5), 33.1, 0.05, 'e^3.5')
w33 = {k: vm(k)/np.sqrt(k) for k in range(2, 34) if vm(k) > 0}
top5 = sorted(w33, key=w33.get, reverse=True)[:5]
assert top5 == [7, 11, 5, 13, 17], top5
close(w33[23], 0.654, 6e-4, 'weight 23')
for k in [29, 31, 25, 27, 32]:
    assert w33[k] < w33[23]
assert w33[17] > w33[3] and w33[19] > w33[3] and w33[23] > w33[3]
d4 = [T/(2*np.pi)*w33[k] for k in [7, 11, 5, 13]]
assert max(d4) - min(d4) < 6, d4
# Answer 21: computed spectrum, five deepest dips at prime-power logs
omega = np.linspace(0.3, 3.5, 6401)
spec = -np.array([np.cos(w*z).sum() for w in omega])
pk, _ = find_peaks(spec, distance=20)
top = pk[np.argsort(spec[pk])[::-1][:5]]
ppw = [k for k in range(2, 34) if vm(k) > 0]
for i in top:
    assert min(abs(omega[i] - np.log(k)) for k in ppw) < 0.005, omega[i]

# Gamma term: digamma ~ log z - 1/(2z)
zc = mpmath.mpc(10, 30)
assert abs(mpmath.digamma(zc) - (mpmath.log(zc) - 1/(2*zc))) < 1e-3

# Duality amplitude sample
rho1 = abs(0.5 + 1j*t1); close(rho1, 14.143566, 1e-6, '|rho1|')
close(t1**2, 199.790451, 1e-5, 't1^2')
close(2*np.sqrt(10)/rho1, 0.4472, 1e-4, 'amp x=10')
close(2*100**0.5/rho1, 1.4141, 1e-4, 'amp x=100')
x = 30.5; zz = qmnt.zeros(100); r = 0.5 + 1j*zz
approx = x - 2*np.sum(np.real(np.exp(r*np.log(x))/r)) - np.log(2*np.pi) - 0.5*np.log(1 - x**-2)
psi_exact = sum(np.log(p)*int(np.floor(np.log(x)/np.log(p))) for p in qmnt.primes_upto(30))
assert abs(approx - psi_exact) < 3.0

# Answer 24 (new problem 24)
t2 = qmnt.zeros(2)[1]; close(t2, 21.022040, 1e-6, 't2')
r2 = abs(0.5 + 1j*t2); close(r2, 21.0280, 1e-4, '|rho2|')
close(2*np.sqrt(1000), 63.2456, 1e-4, '2 sqrt 1000')
close(2*np.sqrt(1000)/r2, 3.008, 1e-3, 'amp rho2 x=1000')
close(10**0.6, 3.981, 1e-3, '10^0.6')
modratio = r2/abs(0.6 + 1j*t2)
assert abs(1 - modratio) < 2e-4, modratio
close(10**0.6*modratio, 3.981, 1e-3, 'beta=0.6 factor')

# Answer 25: Gaussian transform and cutoff
for a, u in [(1.0, 1.0), (2.0, 0.7)]:
    I = quad(lambda rr: np.exp(-rr**2/(2*a*a))*np.cos(rr*u), -np.inf, np.inf)[0]
    close(I/(2*np.pi), a/np.sqrt(2*np.pi)*np.exp(-a*a*u*u/2), 1e-7, 'Gaussian transform')
close(np.real(np.exp(-(0.5j)**2/2)), np.exp(1/8), 1e-12, 'h(i/2) a=1')
close(np.exp(-np.log(2)**2/2), 0.786, 5e-4, 'cutoff k=2')
close(np.exp(-np.log(100)**2/2), 2.5e-5, 1e-6, 'cutoff k=100')

# Problems 2, 3, 4, 7, 8, 13, 14, 17
close(13**-0.5, 0.27735, 1e-5, 'P2a'); close(1/169, 0.005917, 1e-6, 'P2b')
close(2*np.pi/np.log(3), 5.719, 1e-3, 'P3a'); close(2*np.pi/(2*np.log(2)), 4.532, 1e-3, 'P3b')
close(2*np.pi/np.log(13), 2.450, 1e-3, 'P3c')
close(0.263/0.074, 3.55, 0.01, 'P4a'); close(0.296/0.119, 2.49, 0.01, 'P4b')
p3 = 1/3 + 1/36 + 1/243; close(p3, 0.3652263, 1e-7, 'P7 sum')
close(np.sqrt(p3/2), 0.4273326, 1e-7, 'P7 sqrt'); close(np.sqrt(p3/2)/np.pi, 0.1360, 1e-4, 'P7')
tt3 = np.linspace(0, 2*np.pi/np.log(3), 200001)
close(np.sqrt(np.mean(Nfl(tt3, [3])**2)), 0.1360, 1e-4, 'P7 numeric')
xx = 1/np.sqrt(2)
close(-np.log(1 - xx), 1.2279471, 1e-7, 'P8 full'); close(xx + 0.25 + xx**3/3, 1.0749579, 1e-7, 'P8 head')
tail = (-np.log(1 - xx) - (xx + 0.25 + xx**3/3))
close(tail, 0.1529892, 1e-7, 'P8 tail'); close(tail/np.pi, 0.0487, 1e-4, 'P8')
close(sum(1/(m*2**(m/2)) for m in range(4, 400))/np.pi, tail/np.pi, 1e-10, 'P8 direct')
close(0.265/0.198, 1.34, 0.005, 'P13')
close(dNsm(1000.0), 0.8069, 1e-4, 'P14'); close(np.log(1000/(2*np.pi)), 5.0699, 1e-4, 'P14 log')
close(3*np.log(97), 13.7, 0.05, 'P14 max oracle freq')
close(0.074/0.263, 0.281, 0.001, 'P17')
print('all checks passed')
