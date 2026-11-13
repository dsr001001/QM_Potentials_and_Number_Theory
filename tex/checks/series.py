import numpy as np, math
import mpmath as mp
import qmnt

mp.mp.dps = 30
gamma = 0.5772156649015329

# RE1: first zero
t = qmnt.zeros(10)
t1 = float(t[0])
assert abs(t1 - 14.134725) < 1e-6
assert abs(float(t[1]) - 21.022040) < 1e-6

# Sample How far out must you go: N = 31
assert 1/31**2 >= 1e-3 and 1/32**2 < 1e-3
assert abs(math.sqrt(1000) - 31.62) < 0.01

# Example: |e^{i theta}-1| = 2|sin(theta/2)|
th = math.log(2)
assert abs(abs(np.exp(1j*th) - 1) - 2*abs(math.sin(th/2))) < 1e-14

# Geometric r=1/2: S_5 = 1.9375
assert abs(sum(0.5**k for k in range(5)) - 1.9375) < 1e-12
# Archimedes: 1/(1-1/4) = 4/3
assert abs(1/(1 - 0.25) - 4/3) < 1e-12
# complex geometric (i/2)
zsum = 1/(1 - 0.5j)
assert abs(zsum - (0.8 + 0.4j)) < 1e-12

# Sample: ratio 2^{-1/2}
r = 2**-0.5
assert abs(r - 0.707107) < 1e-6
assert abs(1/(1 - r) - 3.414214) < 1e-6
assert abs(2 + math.sqrt(2) - 1/(1 - r)) < 1e-12
assert abs((1 - r) - 0.292893) < 1e-6
assert abs(0.01*(1 - r) - 0.00292893) < 1e-8
assert abs(math.log(0.00292893) - (-5.83317)) < 1e-4
assert abs(math.log(r) - (-0.346574)) < 1e-6
assert abs(5.83317/0.346574 - 16.83) < 0.01
nmin = next(n for n in range(1, 100) if r**n/(1 - r) < 0.01)
assert nmin == 17
assert abs(r**16/(1 - r) - 0.0133) < 1e-4
assert abs(r**17/(1 - r) - 0.0094) < 1e-4

# Sample: bouncing ball
h, g, rr = 1.0, 9.8, 0.5
D = h + 2*h*rr/(1 - rr)
assert abs(D - 3.0) < 1e-12
tau = math.sqrt(2*h/g)
assert abs(tau - 0.451754) < 1e-6
fac = 1 + 2*math.sqrt(rr)/(1 - math.sqrt(rr))
assert abs(fac - 5.828427) < 1e-6
assert abs(tau*fac - 2.633) < 5e-4

# Sample: telescoping with gap, sum 1/(k^2-1) = 3/4
part10 = sum(1/(k**2 - 1) for k in range(2, 11))
assert abs(part10 - 0.654545) < 1e-6
assert abs(0.5*(1.5 - 0.1 - 1/11) - part10) < 1e-12
assert abs(sum(1/(k**2 - 1) for k in range(2, 200001)) - 0.75) < 1e-5

# Harmonic numbers
H = lambda n: math.fsum(1.0/k for k in range(1, n + 1))
assert abs(H(8) - 2.717857) < 1e-6
assert abs(H(10) - 2.928968) < 1e-6
assert abs(math.log(11) - 2.397895) < 1e-6
assert abs(1 + math.log(10) - 3.302585) < 1e-6
assert abs(H(10) - math.log(10) - 0.626383) < 1e-6
# Sample: H_n exceeds 10
assert 2**18 == 262144
assert H(4) > 1 + 2/2  # strict Oresme inequality for K = 2
assert abs(math.exp(10 - gamma) - 12367) < 1.0
assert abs((10 - gamma) - 9.4227843) < 1e-6
assert H(12366) < 10 < H(12367)
assert 21 < 262144/12367 < 22
assert abs(1/(2*12367) - 4e-5) < 1e-6
# slowest infinity: about 1.5e43 terms to pass 100
assert abs(math.exp(100 - gamma)/1.5e43 - 1) < 0.02

# Sample: zeta(2) bracket
S10 = math.fsum(1/k**2 for k in range(1, 11))
assert abs(S10 - 1.549768) < 1e-6
z2 = math.pi**2/6
assert abs(z2 - 1.644934) < 1e-6
lo, hi = S10 + 1/11, S10 + 1/10
assert abs(lo - 1.640677) < 1e-6 and abs(hi - 1.649768) < 1e-6
assert lo <= z2 <= hi
mid = (lo + hi)/2
assert abs(mid - 1.645222) < 1e-6
assert abs((mid - z2) - 0.00029) < 1e-5
assert abs((z2 - S10) - 0.095) < 1e-3

# Sample: sum k^{-1/2} to 1000
S_half = math.fsum(k**-0.5 for k in range(1, 1001))
lo = 2*(math.sqrt(1001) - 1); hi = 1 + 2*(math.sqrt(1000) - 1)
assert abs(lo - 61.277168) < 1e-6 and abs(hi - 62.245553) < 1e-6
assert lo <= S_half <= hi
assert abs(S_half - 61.801) < 5e-4
assert abs(2*1000 + float(mp.zeta(0.5)) - 1998.54) < 0.01

# Checkpoint: tail of k^{-1.5} after 100 terms <= 0.2
assert abs(2/math.sqrt(100) - 0.2) < 1e-12

# Sample: s = 2 + i t1, 100 terms within 0.01
s = 2.0 + 1j*t1
k = np.arange(1, 101)
S100 = np.sum(np.exp(-s*np.log(k)))
zs = complex(mp.zeta(mp.mpc(2.0, t1)))
assert abs(S100 - zs) < 0.01

# spiral at rho_1
s = 0.5 + 1j*t1
k = np.arange(1, 1001)
S = np.cumsum(np.exp(-s*np.log(k)))
main = k**(1 - s)/(1 - s)
assert abs(abs(1 - s) - 14.1436) < 1e-3
assert abs(math.sqrt(1000)/abs(1 - s) - 2.236) < 1e-3
assert abs(abs(S[-1] - main[-1]) - 0.016) < 0.003
assert abs(0.5*1000**-0.5 - 0.0158) < 1e-4

# Sample: log 2 slowly
assert abs(7/12 - 0.583333) < 1e-6
assert abs(math.log(2) - 7/12 - 0.109814) < 1e-6
terms = []
for j in range(1, 200001):
    terms += [1/(2*j - 1), -1/(4*j - 2), -1/(4*j)]
assert abs(math.fsum(terms) - 0.5*math.log(2)) < 1e-5
assert abs(0.5*math.log(2) - 0.346574) < 1e-6

# Sample: log 1.5
x = 1/3
three = x + x**2/2 + x**3/3
assert abs(three - 0.401235) < 1e-6
bnd = x**4/(4*(1 - x))
assert abs(bnd - 0.004630) < 1e-6
assert abs(math.log(1.5) - 0.405465) < 1e-6
err = math.log(1.5) - three
assert abs(err - 0.004230) < 1e-6 and 0 < err <= bnd

# complex log series beyond box (200 terms, within 1e-12)
z = 0.5*np.exp(1j)
L = sum(z**m/m for m in range(1, 200))
assert abs(np.exp(L) - 1/(1 - z)) < 1e-12

# Sample RE2 tail
full = -math.log(1 - r)
assert abs(full - 1.227947) < 1e-6
first3 = r + 0.5/2 + r**3/3
assert abs(first3 - 1.074958) < 1e-6
assert abs(full - first3 - 0.152989) < 1e-6
tail2 = (full - first3)/math.pi
assert abs(tail2 - 0.048698) < 1e-6
crude = (0.25/(1 - r))/(4*math.pi)
assert abs(crude - 0.067924) < 1e-6
assert abs(r/math.pi - 0.225079) < 1e-6
assert abs(tail2/(r/math.pi) - 0.216) < 1e-3
assert abs(2*math.pi/math.log(2) - 9.065) < 1e-3
tt = np.linspace(0, 40, 4001)
w = lambda mm: -sum(np.sin(m*tt*math.log(2))/(m*2**(m/2)) for m in range(1, mm+1))/math.pi
assert np.max(np.abs(w(40) - w(3))) <= tail2 + 1e-12

# Products
assert abs(11/20 - 0.55) < 1e-12
pr = np.prod([1 - 1/k**2 for k in range(2, 11)])
assert abs(pr - 0.55) < 1e-12
primes = [2, 3, 5, 7, 11, 13]
vals = np.cumprod([p**2/(p**2 - 1) for p in primes])
claim = [1.333333, 1.5, 1.5625, 1.595052, 1.608344, 1.617918]
ratios = [0.8106, 0.9119, 0.9499, 0.9697, 0.9778, 0.9836]
for v, c, q in zip(vals, claim, ratios):
    assert abs(v - c) < 1e-6
    assert abs(v/z2 - q) < 1e-4
assert abs(1225/768 - 1.595052) < 1e-6

# Pitfall: complex log
zz = np.exp(2j*np.pi/3)
assert abs((np.log(zz) + np.log(zz)) - np.log(zz*zz) - 2j*np.pi) < 1e-12

# Answers
assert abs(40/27 - 1.481481) < 1e-6                       # P1
assert abs(3**-0.5 - 0.577350) < 1e-6                     # P4
assert abs(math.log(3) - 1.098612) < 1e-6
S5 = math.fsum(1/k**3 for k in range(1, 6))               # P7
assert abs(S5 - 1.185662) < 1e-6
assert abs(S5 + 1/72 - 1.199551) < 1e-6 and abs(S5 + 1/50 - 1.205662) < 1e-6
z3 = float(mp.zeta(3)); assert abs(z3 - 1.202057) < 1e-6
assert S5 + 1/72 <= z3 <= S5 + 1/50
alt2 = math.fsum((-1)**(k+1)/k**2 for k in range(1, 200001))  # P9
assert abs(alt2 - math.pi**2/12) < 1e-9
assert abs(math.pi**2/12 - 0.822467) < 1e-6
b = lambda n: 2.0**-n/(n + 1)                              # P11
assert b(15) > 1e-6 and b(16) < 1e-6
assert abs(b(15) - 1.9e-6) < 1e-7 and abs(b(16) - 9.0e-7) < 1e-7
assert abs(math.fsum(0.5**m/m for m in range(1, 17)) - math.log(2)) < 1e-6
assert abs(1 + z2 - 2.644934) < 1e-6                       # P13
assert abs(math.exp(z2) - 5.180668) < 1e-6
assert abs(math.sinh(math.pi)/math.pi - 3.676078) < 1e-6
r3 = 3**-0.5                                               # P15
full3 = -math.log(1 - r3); f3 = r3 + (1/3)/2 + r3**3/3
assert abs(full3 - 0.861212) < 1e-6 and abs(f3 - 0.808167) < 1e-6
assert abs(full3 - f3 - 0.053045) < 1e-6
assert abs((full3 - f3)/math.pi - 0.016885) < 1e-6
assert abs((1/9)/(1 - r3)/(4*math.pi) - 0.020920) < 1e-6
for a, q in [(0.198, 0.753), (0.116, 0.441), (0.074, 0.281), (0.043, 0.163)]:  # P17
    assert abs(a/0.263 - q) < 1e-3
H6 = math.fsum(1.0/k for k in range(1, 10**6 + 1))          # P19
assert abs(H6 - 14.392727) < 1e-6
assert abs(math.log(1e6) - 13.815511) < 1e-6
assert abs(H6 - (math.log(1e6) + gamma + 0.5e-6)) < 1e-11
# P21 oracle rows of Table V
n = np.arange(401, 501)
rms0, _ = qmnt.score(qmnt.smooth_zeros(n), n)
assert abs(rms0 - 0.263) < 1e-3
for kk, (rm, cr) in zip([1, 3, 6, 25], [(0.198, 0.66), (0.116, 0.90), (0.074, 0.96), (0.043, 0.99)]):
    tp = qmnt.oracle_zeros(n, qmnt.first_primes(kk), mmax=3)
    a, c = qmnt.score(tp, n)
    assert abs(a - rm) < 1.5e-3 and abs(c - cr) < 0.015
# P23 spiral sizes
kk = np.arange(1, 10001)
SS = np.cumsum(np.exp(-(0.5 + 1j*t1)*np.log(kk)))
for N, pred in [(100, 0.707), (1000, 2.236), (10000, 7.070)]:
    assert abs(math.sqrt(N)/abs(0.5 - 1j*t1) - pred) < 1e-3
    assert abs(abs(SS[N-1]) - pred) < 0.6/math.sqrt(N) + 1e-3
# P25
assert abs(1.5*math.log(2) - 1.039721) < 1e-6
# P27
z32 = float(mp.zeta(1.5)); assert abs(z32 - 2.612375) < 1e-6
bound27 = (1/(3*math.pi))*(1/(1 - r))*(z32 - 1)
assert abs(bound27 - 0.584) < 1e-3
# P29
assert [3*q for q in [1, 3, 6, 25, 300]] == [3, 9, 18, 75, 900]

# sum rule numerical instance (Eq. pair)
rng = np.random.default_rng(0)
Aab = rng.normal(size=(6, 6)); Aab = (Aab + Aab.T)/2
Ev = np.sort(rng.normal(size=6))*10
T = sum(-Aab[a, m]**2/(Ev[a] - Ev[m]) for a in range(6) for m in range(6) if m != a)
assert abs(T) < 1e-12
print("all checks passed")
