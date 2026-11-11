import numpy as np, mpmath, qmnt
from scipy.optimize import brentq
from scipy.special import lambertw

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
            s += np.log(p)*np.cos(m*t*np.log(p))/p**(m/2)
    return -s/np.pi

def smooth_zero(n):
    return brentq(lambda t: float(Nsm(t)) - (n - 0.5), 5.0, 1e5)

t1 = 14.134725141734695
close(qmnt.zeros(1)[0], t1, 1e-9, "t1")

# Sample: log zeta(2) from primes 2,3,5,7
lz2 = np.log(np.pi**2/6); close(lz2, 0.4977003, 1e-6, "log zeta(2)")
terms = [np.log(p*p/(p*p-1.0)) for p in [2, 3, 5, 7]]
for v, c in zip(terms, [0.2876821, 0.1177830, 0.0408220, 0.0206193]):
    close(v, c, 1e-6, "SP1 terms")
close(sum(terms), 0.4669064, 1e-6, "SP1 sum")
close(sum(terms)/lz2, 0.938, 0.0005, "SP1 fraction")
close(lz2 - sum(terms), 0.0307943, 1e-6, "SP1 remainder")
# full prime sum agrees with log zeta(2)
pp = np.asarray(qmnt.primes_upto(10**6), float)
close(np.sum(-np.log(1 - pp**-2.0)), lz2, 1e-5, "Euler log series")

# Sample: 3^{-s} at s=1/2+10i
z3 = 3.0**(-(0.5 + 10j))
close(z3.real, -0.005457, 2e-6, "SP2 real"); close(z3.imag, 0.577324, 2e-6, "SP2 imag")
close(-np.sin(10*np.log(3))/np.sqrt(3)/np.pi, 0.18377, 2e-5, "SP2 wave")

# Amplitude table
tab = [(2, 1, 0.6931, 9.0647, 0.7071), (3, 1, 1.0986, 5.7192, 0.5774), (2, 2, 1.3863, 4.5324, 0.2500),
       (5, 1, 1.6094, 3.9040, 0.4472), (7, 1, 1.9459, 3.2289, 0.3780), (2, 3, 2.0794, 3.0216, 0.1179),
       (3, 2, 2.1972, 2.8596, 0.1667), (11, 1, 2.3979, 2.6203, 0.3015), (13, 1, 2.5649, 2.4496, 0.2774)]
for p, m, w, per, amp in tab:
    k = p**m
    close(np.log(k), w, 6e-5, "tab omega"); close(2*np.pi/np.log(k), per, 6e-5, "tab period")
    close(1/(m*np.sqrt(k)), amp, 6e-5, "tab amp")
close(np.log(25), 3.2189, 6e-5, "checkpoint 25")

# RE2 periods and rms
close(2*np.pi/np.log(2), 9.0647, 1e-4, "RE2 period")
close(2*np.pi/(2*np.log(2)), 4.5324, 1e-4, "RE2 period 2")
close(2*np.pi/(3*np.log(2)), 3.0216, 1e-4, "RE2 period 3")
ms = sum(1/(m*m*2**m) for m in (1, 2, 3)); close(ms, 0.5763889, 1e-7, "RE2 sum")
re2rms = np.sqrt(ms/2)/np.pi; close(re2rms, 0.1709, 1e-4, "RE2 rms")
tt = np.linspace(0, 2*np.pi/np.log(2), 200001)
close(np.sqrt(np.mean(Nfl(tt, [2])**2)), re2rms, 1e-4, "RE2 rms numeric")
# probe null: rms of N_fl^{p=2}(s_n), n=401..500 vs paper 0.1708
sn = np.array([smooth_zero(n) for n in range(401, 501)])
close(np.sqrt(np.mean(Nfl(sn, [2])**2)), 0.1708, 0.003, "probe null")

# Sum of 1/sqrt(p) for p<=100
p100 = np.asarray(qmnt.primes_upto(100), float)
assert len(p100) == 25
close(np.sum(p100**-0.5), 5.5365, 2e-4, "sum p^-1/2")
close(np.sum(p100**-0.5)/np.pi, 1.76, 0.005, "max m=1 amplitude")

# Sample: N_fl at t1, N_sm(t1), S(t1)
close(float(Nfl(t1, [2])), 0.0617, 3e-4, "SP Nfl(t1)")
close(float(Nsm(t1)), 0.44928, 2e-5, "SP Nsm(t1)")
close(0.5 - float(Nsm(t1)), 0.05072, 2e-5, "SP S(t1)")
close(1/(48*np.pi*t1), 0.0005, 5e-5, "1/t correction")
close(t1/(2*np.pi), 2.2496113, 1e-6, "t1/2pi")

# Sample: predicting t1 with p=2
s1 = smooth_zero(1); close(s1, 14.5214, 2e-4, "s1")
g1 = 2*np.pi/np.log(s1/(2*np.pi)); close(g1, 7.500, 2e-3, "gap1")
close(np.log(s1/(2*np.pi)), 0.837746, 2e-5, "log s1/2pi")
nf = float(Nfl(s1, [2])); close(nf, 0.09350, 2e-4, "Nfl(s1)")
dnf = dNfl(s1, [2]); close(dnf, 0.0667, 3e-4, "Nfl'(s1)")
close(s1 - g1*nf, 13.820, 3e-3, "crude first order")
close(s1 - g1*nf*1.0 + 0.0, 13.820, 3e-3, "crude first order 2")
close(s1 - nf/(1/g1 + dnf), 14.054, 3e-3, "better first order")
f = lambda t: float(Nsm(t) + Nfl(t, [2])) - 0.5
assert f(13.8202) < 0 < f(14.1347)
close(f(13.8202), -0.0635, 3e-4, "f(13.82)"); close(f(14.1347), 0.0110, 3e-4, "f(14.135)")
close(f(14.0883), 0.00036, 2e-4, "f(14.088)")
root = brentq(f, 13.8, 14.2); close(root, 14.087, 2e-3, "root t1")
close((t1 - s1)/g1, -0.0516, 5e-4, "true fluct 1")
close((root - s1)/g1, -0.0580, 5e-4, "pred fluct 1")
close(abs((root - s1)/g1 - (t1 - s1)/g1), 0.0064, 5e-4, "oracle error 1")
close(dNsm(t1), 0.129, 1e-3, "slope at t1")

# Uniqueness threshold P={2}
B2 = np.log(2)*(1/np.sqrt(2) + 0.5 + 1/(2*np.sqrt(2)))/np.pi
close(B2, 0.34434, 1e-5, "bound slope p=2")
close(2*np.pi*B2, 2.16353, 1e-5, "log threshold")
close(np.exp(2*np.pi*B2), 8.7018, 1e-3, "t/2pi threshold")
close(2*np.pi*np.exp(2*np.pi*B2), 54.7, 0.05, "threshold 54.7")
# Problem 9: P={2,3}
B23 = (np.log(2)*(1/np.sqrt(2)+0.5+1/(2*np.sqrt(2))) + np.log(3)*(1/np.sqrt(3)+1/3+1/(3*np.sqrt(3))))/np.pi
close(B23, 0.73010, 1e-5, "P9 bound"); close(2*np.pi*B23, 4.58736, 1e-5, "P9 log")
close(np.exp(2*np.pi*B23), 98.235, 0.01, "P9 t/2pi")
close(2*np.pi*np.exp(2*np.pi*B23), 617.2, 0.1, "P9 threshold")

# Lambert W formula
for n in [1, 10, 100, 1000]:
    A = n - 11/8
    sW = (2*np.pi*A/lambertw(A/np.e, 0)).real
    close(sW, smooth_zero(n), 1e-8, "Lambert s_n")
close(lambertw(-3/(8*np.e), 0).real, -0.16226, 1e-5, "W(-3/8e)")
close(-3/(8*np.e), -0.13795, 1e-5, "A/e n=1")
close((2*np.pi*(-0.375)/lambertw(-3/(8*np.e), 0)).real, 14.521, 1e-3, "s1 via W")

# Table V reproduction (rows 401-500 and 501-1000)
z = qmnt.zeros(1000)
n1 = np.arange(401, 501); n2 = np.arange(501, 1001)
close(qmnt.rms(qmnt.fluct(z[n1-1], n1)), 0.2631, 0.001, "null 401-500")
close(qmnt.rms(qmnt.fluct(z[n2-1], n2)), 0.270, 0.0015, "null 501-1000")
rows = {1: [(0.198, 0.66), (0.208, 0.64)], 3: [(0.116, 0.90), (0.127, 0.88)],
        6: [(0.074, 0.96), (0.083, 0.95)], 25: [(0.043, 0.99), (0.050, 0.99)]}
for k, vals in rows.items():
    P = qmnt.first_primes(k)
    for nn, (r, c) in zip([n1, n2], vals):
        rr, cc = qmnt.score(qmnt.oracle_zeros(nn, P, mmax=3), nn)
        close(rr, r, 0.0015, ("TableV rms", k)); close(cc, c, 0.011, ("TableV corr", k))
assert list(qmnt.first_primes(6)) == [2, 3, 5, 7, 11, 13]
assert int(qmnt.first_primes(25)[-1]) == 97

# Landau sample
T = z[-1]; close(T, 1419.42248, 1e-4, "t1000")
close(T/(2*np.pi), 225.908, 1e-3, "T/2pi")
dips = {2: 110.7, 3: 143.3, 4: 78.3, 5: 162.6, 7: 166.2}
lam = {2: np.log(2), 3: np.log(3), 4: np.log(2), 5: np.log(5), 7: np.log(7)}
for k, d in dips.items():
    close(T/(2*np.pi)*lam[k]/np.sqrt(k), d, 0.06, ("Landau dip", k))
close(np.log(2)/np.sqrt(2), 0.490129, 1e-6, "w2"); close(np.log(3)/np.sqrt(3), 0.634285, 1e-6, "w3")
close(np.log(2)/2, 0.346574, 1e-6, "w4")
Ssum = np.sum(np.cos(np.log(2)*z))
assert abs(Ssum + 110.7) < 0.3*110.7, ("Landau numeric", Ssum)
close(np.log(T), 7.26, 0.01, "log T ~ 7")
# Problem 11: weights and argmax among k<=20
def vm(k):
    for p in [2, 3, 5, 7, 11, 13, 17, 19]:
        q = p
        while q <= k:
            if q == k: return np.log(p)
            q *= p
    return 0.0
wts = {k: vm(k)/np.sqrt(k) for k in range(2, 21) if vm(k) > 0}
assert max(wts, key=wts.get) == 7
for k, c in [(2, 0.490), (3, 0.634), (4, 0.347), (5, 0.720), (7, 0.735), (8, 0.245), (9, 0.366),
             (11, 0.723), (13, 0.711), (16, 0.173), (17, 0.687), (19, 0.676)]:
    close(wts[k], c, 6e-4, ("P11 weight", k))
close(np.exp(2), 7.4, 0.05, "peak of log x/sqrt x")

# Duality amplitude sample / problem 24
rho1 = abs(0.5 + 1j*t1); close(rho1, 14.143566, 1e-6, "|rho1|")
close(t1**2, 199.790451, 1e-5, "t1^2")
close(2*np.sqrt(10)/rho1, 0.4472, 1e-4, "amp x=10")
close(2*100**0.5/rho1, 1.4141, 1e-4, "amp x=100")
# psi explicit formula sanity at x=30.5 with 100 zero pairs (rough)
x = 30.5; zz = qmnt.zeros(100); r = 0.5 + 1j*zz
approx = x - 2*np.sum(np.real(np.exp(r*np.log(x))/r)) - np.log(2*np.pi) - 0.5*np.log(1 - x**-2)
psi_exact = sum(np.log(p)*int(np.floor(np.log(x)/np.log(p))) for p in qmnt.primes_upto(30))
assert abs(approx - psi_exact) < 3.0

# Problems 2, 3, 4, 7, 8, 13, 14, 17
close(13**-0.5, 0.27735, 1e-5, "P2a"); close(1/169, 0.005917, 1e-6, "P2b")
close(2*np.pi/np.log(3), 5.719, 1e-3, "P3a"); close(2*np.pi/(2*np.log(2)), 4.532, 1e-3, "P3b")
close(2*np.pi/np.log(13), 2.450, 1e-3, "P3c")
close(0.263/0.074, 3.55, 0.01, "P4a"); close(0.296/0.119, 2.49, 0.01, "P4b")
p3 = 1/3 + 1/36 + 1/243; close(p3, 0.3652263, 1e-7, "P7 sum")
close(np.sqrt(p3/2), 0.4273326, 1e-7, "P7 sqrt"); close(np.sqrt(p3/2)/np.pi, 0.1360, 1e-4, "P7")
tt3 = np.linspace(0, 2*np.pi/np.log(3), 200001)
close(np.sqrt(np.mean(Nfl(tt3, [3])**2)), 0.1360, 1e-4, "P7 numeric")
xx = 1/np.sqrt(2)
close(-np.log(1 - xx), 1.2279471, 1e-7, "P8 full"); close(xx + 0.25 + xx**3/3, 1.0749579, 1e-7, "P8 head")
tail = (-np.log(1 - xx) - (xx + 0.25 + xx**3/3))
close(tail, 0.1529892, 1e-7, "P8 tail"); close(tail/np.pi, 0.0487, 1e-4, "P8")
close(sum(1/(m*2**(m/2)) for m in range(4, 400))/np.pi, tail/np.pi, 1e-10, "P8 direct")
close(0.265/0.198, 1.34, 0.005, "P13")
close(dNsm(1000.0), 0.8069, 1e-4, "P14"); close(np.log(1000/(2*np.pi)), 5.0699, 1e-4, "P14 log")
close(3*np.log(97), 13.7, 0.05, "P14 max oracle freq")
close(0.074/0.263, 0.281, 0.001, "P17")
print("all checks passed")
