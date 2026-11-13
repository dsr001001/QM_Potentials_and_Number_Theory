import numpy as np
import math
import cmath
from math import factorial

pi = math.pi
log2 = math.log(2.0)

def get_zeros(n):
    try:
        import qmnt
        return np.asarray(qmnt.zeros(n), dtype=float)
    except Exception:
        import mpmath
        return np.array([float(mpmath.zetazero(k).imag) for k in range(1, n + 1)])

# ---------- RE1: first ten zeros ----------
RE1 = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
tz = get_zeros(10)
assert np.allclose(tz, RE1, atol=1e-6)
t1 = 14.134725
t2 = 21.022040

# ---------- Sample: multiplying ----------
z = 3 + 2j; w = 1 - 4j
assert abs(z * w - (11 - 10j)) < 1e-12
assert abs((1j) ** 27 - (-1j)) < 1e-12

# ---------- Distributive law (text), numerical instance ----------
zz, ww, uu = 1.3 - 0.7j, -2.1 + 0.4j, 0.9 + 3.3j
assert abs(zz * (ww + uu) - (zz * ww + zz * uu)) < 1e-12

# ---------- Sample: dividing ----------
q = z / w
assert abs(q - (-5 + 14j) / 17) < 1e-12
assert abs(q.real - (-0.294118)) < 1e-6 and abs(q.imag - 0.823529) < 1e-6

# ---------- Sample: modulus and argument of rho_1 ----------
rho1 = 0.5 + 1j * t1
assert abs(t1 ** 2 - 199.790451) < 1e-6
assert abs(abs(rho1) ** 2 - 200.040451) < 1e-6
assert abs(abs(rho1) - 14.1436) < 1e-4
assert abs(math.atan(0.5 / t1) - 0.0353592) < 1e-7
assert abs(np.angle(rho1) - 1.53544) < 1e-5
assert abs(math.degrees(np.angle(rho1)) - 87.974) < 1e-3

# ---------- Sample: (1+i)^8 ----------
assert abs((1 + 1j) ** 8 - 16) < 1e-12

# ---------- Checkpoint: arg sum 4 - 2pi ----------
assert abs((4 - 2 * pi) - (-2.2832)) < 1e-4
assert abs(np.angle(np.exp(3j) * np.exp(1j)) - (4 - 2 * pi)) < 1e-12

# ---------- e partial sum ----------
assert abs(sum(1 / factorial(n) for n in range(6)) - 2.716667) < 1e-6
assert abs(math.e - 2.718282) < 1e-6

# ---------- Sample: partial sums for e^{i pi/3} ----------
theta = pi / 3
terms = [(1j * theta) ** n / factorial(n) for n in range(9)]
claimed = [(1.00000, 0.00000), (1.00000, 1.04720), (0.45169, 1.04720),
           (0.45169, 0.85580), (0.50180, 0.85580), (0.50180, 0.86630),
           (0.49996, 0.86630), (0.49996, 0.86602), (0.50000, 0.86602)]
S = 0
for n in range(9):
    S += terms[n]
    assert abs(S.real - claimed[n][0]) < 1e-5, n
    assert abs(S.imag - claimed[n][1]) < 1e-5, n
assert abs(theta - 1.047198) < 1e-6
assert abs(-theta ** 2 / 2 - (-0.548311)) < 1e-6
assert abs(theta ** 3 / 6 - 0.191397) < 1e-6
assert abs(theta ** 4 / 24 - 0.050108) < 1e-6
assert abs(theta ** 5 / 120 - 0.010494) < 1e-6
assert abs(theta ** 6 / 720 - 0.001832) < 1e-6
assert abs(theta ** 7 / 5040 - 0.000274) < 1e-6
assert abs(theta ** 8 / 40320 - 0.000036) < 1e-6
assert abs(math.sqrt(3) / 2 - 0.866025) < 1e-6
assert abs(pi ** 2 / 2 - 4.9) < 0.04
assert abs(np.exp(1j * pi) + 1) < 1e-15

# ---------- Checkpoint: |e^{2+3i}| ----------
assert abs(abs(np.exp(2 + 3j)) - 7.389) < 1e-3

# ---------- Derivative of e^{kt} (text), numerical instance ----------
k = 0.3 + 2.1j; tt0 = 0.8; h = 1e-6
num = (np.exp(k * (tt0 + h)) - np.exp(k * (tt0 - h))) / (2 * h)
assert abs(num - k * np.exp(k * tt0)) < 1e-6
al, om0 = k.real, k.imag
up = al * math.exp(al * tt0) * math.cos(om0 * tt0) - om0 * math.exp(al * tt0) * math.sin(om0 * tt0)
vp = al * math.exp(al * tt0) * math.sin(om0 * tt0) + om0 * math.exp(al * tt0) * math.cos(om0 * tt0)
assert abs(complex(up, vp) - k * np.exp(k * tt0)) < 1e-12

# ---------- Sample: phasor ----------
ph = 3 - 4j
assert abs(abs(ph) - 5) < 1e-12
assert abs(np.angle(ph) - (-0.927295)) < 1e-6
tt = np.linspace(0, 10, 101)
assert np.allclose(3 * np.cos(tt) + 4 * np.sin(tt), 5 * np.cos(tt - 0.927295218), atol=1e-8)

# ---------- Sample: beats ----------
assert abs(math.log(6) / 2 - 0.895880) < 1e-6
assert abs(math.log(1.5) / 2 - 0.202733) < 1e-6
assert abs(2 * pi / math.log(1.5) - 15.50) < 0.01
tt = np.linspace(0, 50, 501)
lhs = np.sin(tt * log2) + np.sin(tt * math.log(3))
rhs = 2 * np.sin(tt * math.log(6) / 2) * np.cos(tt * math.log(1.5) / 2)
assert np.allclose(lhs, rhs, atol=1e-12)

# ---------- Sample: logs and i^i ----------
assert abs(cmath.log(-2) - complex(0.693147, 3.141593)) < 1e-6
assert abs(cmath.log(1 + 1j) - complex(0.346574, 0.785398)) < 1e-6
assert abs((1j) ** (1j) - 0.207880) < 1e-6
assert abs(math.exp(-pi / 2) - 0.207880) < 1e-6

# ---------- Table: primes on the critical line ----------
tab = {2: (0.7071, 0.6931, 9.0647), 3: (0.5774, 1.0986, 5.7192),
       5: (0.4472, 1.6094, 3.9040), 7: (0.3780, 1.9459, 3.2289),
       11: (0.3015, 2.3979, 2.6203), 13: (0.2774, 2.5649, 2.4496)}
for p, (r, lp, per) in tab.items():
    assert abs(p ** -0.5 - r) < 1e-4, p
    assert abs(math.log(p) - lp) < 1e-4, p
    assert abs(2 * pi / math.log(p) - per) < 1e-4, p
assert abs(2 * pi / log2 - 9.065) < 1e-3
assert abs(math.log(2) - 0.693) < 1e-3
assert abs(1 / math.sqrt(3) - 0.5774) < 1e-4

# ---------- (p^{-s})^m = (p^m)^{-s} instance ----------
sx = 0.5 + 7.3j
for p in (2, 3):
    for m in (1, 2, 3):
        assert abs((p ** (-sx)) ** m - (p ** m) ** (-sx)) < 1e-12

# ---------- Sample: 2^{-rho_1} ----------
s1 = 0.5 + 1j * t1
v = 2 ** (-s1)
assert abs(abs(v) - 0.70711) < 1e-5
assert abs(-t1 * log2 - (-9.79744)) < 1e-5
assert abs(4 * pi - 12.56637) < 1e-5
assert abs(np.angle(v) - 2.76893) < 1e-5
assert abs(math.cos(2.768926) - (-0.931360)) < 2e-6
assert abs(math.sin(2.768926) - 0.364101) < 2e-6
assert abs(v.real - (-0.65857)) < 1e-4 and abs(v.imag - 0.25746) < 1e-4

# ---------- Quadratic formula for complex coefficients (text), instance ----------
a_, b_, c_ = 1.5 - 0.5j, -2 + 1j, 0.7 + 3j
D = b_ ** 2 - 4 * a_ * c_
wq = cmath.sqrt(D)
for sgn in (1, -1):
    zr = (-b_ + sgn * wq) / (2 * a_)
    assert abs(a_ * zr ** 2 + b_ * zr + c_) < 1e-12

# ---------- Sample: quadratic ----------
c = 0.25 + t1 ** 2
assert abs(c - 200.0405) < 1e-4
roots = np.roots([1, -1, c])
assert np.allclose(sorted(roots, key=lambda r: r.imag), [0.5 - 1j * t1, 0.5 + 1j * t1], atol=1e-9)

# ---------- Sample: zeros of 1 - 2^{-s} ----------
for kk in range(-3, 4):
    assert abs(1 - 2 ** (-(2j * pi * kk / log2))) < 1e-12
assert abs(2 * 2 * pi / log2 - 18.1294) < 1e-4
tt = np.linspace(-50, 50, 20001)
assert np.min(np.abs(1 - 2.0 ** (-0.5 - 1j * tt))) >= 1 - 2 ** -0.5 - 1e-12

# ---------- Sample: argument of products (incl. tan 75 deg) ----------
assert abs(5 * pi / 12 - 1.308997) < 1e-6
prod = (1 + 1j) * (math.sqrt(3) + 1j)
assert abs(prod.real - 0.732051) < 1e-6 and abs(prod.imag - 2.732051) < 1e-6
assert abs(np.angle(prod) - 5 * pi / 12) < 1e-12
assert abs((1 + 1 / math.sqrt(3)) / (1 - 1 / math.sqrt(3)) - (2 + math.sqrt(3))) < 1e-12
assert abs((math.sqrt(3) + 1) ** 2 / 2 - (2 + math.sqrt(3))) < 1e-12
assert abs(2 + math.sqrt(3) - 3.732051) < 1e-6
assert abs(math.tan(math.radians(75)) - (2 + math.sqrt(3))) < 1e-12
assert abs(math.atan(3.732051) - 1.308997) < 1e-6
tstar = pi / (2 * log2)
assert abs(tstar - 2.266180) < 1e-6
val = 1 - 2 ** (-(0.5 + 1j * tstar))
assert abs(val - (1 + 1j / math.sqrt(2))) < 1e-12
assert abs(np.angle(val) - 0.615480) < 1e-6
three = sum(math.sin(m * tstar * log2) / (m * 2 ** (m / 2)) for m in (1, 2, 3))
assert abs(-0.615480 / pi - (-0.195913)) < 1e-6
assert abs(-three / pi - (-0.187566)) < 1e-6

# ---------- Figure arg vs RE2; bound 0.0487 (Problem 23) ----------
tt = np.linspace(0, 40, 40001)
x = 2.0 ** (-0.5 - 1j * tt)
phase = -np.angle(1 - x) / np.pi
re2 = -sum(np.sin(m * tt * log2) / (m * 2 ** (m / 2)) for m in (1, 2, 3)) / np.pi
maxdiff = np.max(np.abs(phase - re2))
assert maxdiff < 0.05
tail = sum(1 / (pi * m * 2 ** (m / 2)) for m in range(4, 201))
assert abs(tail - 0.0487) < 1e-3
assert maxdiff <= tail + 1e-12
full = sum(np.sin(m * tt * log2) / (m * 2 ** (m / 2)) for m in range(1, 400))
assert np.max(np.abs(np.angle(1 - x) - full)) < 1e-10
# principal-branch bound -log(1-1/sqrt2) ~ 1.228 < pi
bnd = -math.log(1 - 2 ** -0.5)
assert abs(bnd - 1.228) < 1e-3 and bnd < pi
# e^{S(x)} = 1 - x for complex |x|<1 (beyond box), instance
xs = 0.5 * np.exp(1.1j)
Sx = -sum(xs ** m / m for m in range(1, 200))
assert abs(np.exp(Sx) - (1 - xs)) < 1e-12

# ---------- Paper connection: e^8 ~ 2981 ----------
assert abs(math.exp(8) - 2981) < 1

# ---------- Problem 1 ----------
z = 2 - 3j; w = -1 + 5j
assert abs(z + w - (1 + 2j)) < 1e-12
assert abs(z * w - (13 + 13j)) < 1e-12
assert abs(z / w - (-17 - 7j) / 26) < 1e-12
assert abs((z / w).real - (-0.653846)) < 1e-6 and abs((z / w).imag - (-0.269231)) < 1e-6

# ---------- Problem 2 ----------
assert abs((1j) ** 7 - (-1j)) < 1e-12
assert abs((1j) ** 10 - (-1)) < 1e-12
assert abs(1 / 1j - (-1j)) < 1e-12
assert abs((1j) ** 2027 - (-1j)) < 1e-9

# ---------- Problem 3 ----------
assert abs(abs(-math.sqrt(3) + 1j) - 2) < 1e-12
assert abs(np.angle(-math.sqrt(3) + 1j) - 5 * pi / 6) < 1e-12
assert abs(np.angle(-4 + 0j) - pi) < 1e-12
assert abs(np.angle(-2j) - (-pi / 2)) < 1e-12
assert abs(abs(3 - 3j) - 4.2426) < 1e-4 and abs(np.angle(3 - 3j) + pi / 4) < 1e-12

# ---------- Problem 4 ----------
assert abs(2 * np.exp(1j * pi / 4) - (math.sqrt(2) + 1j * math.sqrt(2))) < 1e-12
assert abs(5 * np.exp(-1j * pi / 2) - (-5j)) < 1e-12
assert abs(np.exp(3j * pi) - (-1)) < 1e-12
assert abs(np.angle(-1 - 1j) + 3 * pi / 4) < 1e-12

# ---------- Problem 6 ----------
rho2 = 0.5 + 1j * t2
assert abs(t2 ** 2 - 441.926166) < 1e-5
assert abs(abs(rho2) - 21.0280) < 1e-4
assert abs(math.atan(0.5 / t2) - 0.023780) < 1e-6
assert abs(np.angle(rho2) - 1.547016) < 1e-6
assert abs(math.degrees(np.angle(rho2)) - 88.64) < 0.01

# ---------- Problem 7 ----------
assert abs(np.exp(1j * pi) - (-1)) < 1e-12
assert abs(np.exp(1j * pi / 2) - 1j) < 1e-12
assert abs(np.exp(2j * pi) - 1) < 1e-12
assert abs(np.exp(-1j * pi / 4) - (0.70711 - 0.70711j)) < 1e-5

# ---------- Problem 8 ----------
z8 = 5 + 12j
assert z8.conjugate() == 5 - 12j
assert abs(abs(z8) - 13) < 1e-12
assert abs(1 / z8 - (5 - 12j) / 169) < 1e-15
assert abs((1 / z8).real - 0.029586) < 1e-6 and abs((1 / z8).imag - (-0.071006)) < 1e-6
assert abs(abs(1 / z8) - 0.076923) < 1e-6

# ---------- Problem 9 ----------
z, w, u = 1 + 1j, 2 - 1j, 3j
assert abs(z * w - (3 + 1j)) < 1e-12
assert abs((z * w) * u - (-3 + 9j)) < 1e-12
assert abs(w * u - (3 + 6j)) < 1e-12
assert abs(z * (w * u) - (-3 + 9j)) < 1e-12
# general expansion of associativity, random instance
rng = np.random.default_rng(0)
a, b, c_, d, e, f = rng.normal(size=6)
re_l = a * c_ * e - b * d * e - a * d * f - b * c_ * f
im_l = a * c_ * f - b * d * f + a * d * e + b * c_ * e
lhs = ((a + 1j * b) * (c_ + 1j * d)) * (e + 1j * f)
assert abs(lhs - complex(re_l, im_l)) < 1e-12

# ---------- Problem 11 ----------
th = 0.37
assert abs(math.cos(3 * th) - (4 * math.cos(th) ** 3 - 3 * math.cos(th))) < 1e-12
assert abs(math.sin(3 * th) - (3 * math.sin(th) - 4 * math.sin(th) ** 3)) < 1e-12

# ---------- Problem 13 ----------
tt = np.linspace(0, 2 * pi, 100001)[:-1]
assert abs(np.mean(np.cos(tt) ** 2) - 0.5) < 1e-9

# ---------- Problem 15 ----------
assert abs(cmath.log(-math.e) - complex(1, pi)) < 1e-12
assert abs(cmath.log(2j) - complex(0.69315, 1.57080)) < 1e-5
assert abs((1j) ** (1j) - 0.20788) < 1e-5
v2i = 2 ** (1j)
assert abs(v2i.real - 0.76924) < 1e-5 and abs(v2i.imag - 0.63896) < 1e-5

# ---------- Problem 17 ----------
for p in (2, 3, 5):
    for m in (1, 2, 3):
        tt0 = 7.3
        lhs = math.sin(m * tt0 * math.log(p)) / (m * p ** (m / 2))
        rhs = -(p ** (-m * (0.5 + 1j * tt0))).imag / m
        assert abs(lhs - rhs) < 1e-12

# ---------- Problem 19 ----------
for kk in range(-2, 3):
    assert abs(1 - 3 ** (-(2j * pi * kk / math.log(3)))) < 1e-12
    sk = -1j * pi * (2 * kk + 1) / log2
    assert abs(1 + 2 ** (-sk)) < 1e-12
assert abs(2 * pi / math.log(3) - 5.7192) < 1e-4
assert abs(pi / log2 - 4.5324) < 1e-4

# ---------- Problem 22 ----------
for tn in tz:
    assert abs(abs(2 ** (-(0.5 + 1j * tn))) - 2 ** -0.5) < 1e-12

# ---------- Problem 24 ----------
Gam = 1.0
assert abs(2 * math.log(2) / Gam - 1.386294) < 1e-6
assert abs(math.log(2) / Gam - 0.693147) < 1e-6
E = 100.0; tq = 2 * math.log(2)
assert abs(abs(np.exp(-1j * (E - 0.5j * Gam) * tq)) - 0.5) < 1e-12

# ---------- Problem 25 ----------
for tn in tz:
    rr = 0.5 + 1j * tn
    assert abs(rr * (1 - rr) - (0.25 + tn ** 2)) < 1e-9
assert abs((0.5 + 1j * t1) * (0.5 - 1j * t1) - 200.040451) < 1e-6

# ---------- Problem 26 ----------
R, om = 0.5, 4.0
assert abs(om * R - 2.0) < 1e-12 and abs(om ** 2 * R - 8.0) < 1e-12

# ---------- Problem 27 ----------
thv = np.linspace(0, 3, 31)
Ev = np.cos(thv) + 1j * np.sin(thv)
assert np.allclose(Ev * np.conj(Ev), 1.0)

# ---------- Problem 28 (numbered 27 before reordering: geometric sum) ----------
n, th = 5, 0.7
lhs = sum(math.cos(kk * th) for kk in range(n))
rhs = math.sin(n * th / 2) * math.cos((n - 1) * th / 2) / math.sin(th / 2)
assert abs(lhs - rhs) < 1e-12

# ---------- Problem 29 ----------
tt = np.linspace(0, 2 * pi, 20001)
for r, expected in [(2 ** -0.5, 0.0), (2.0, -2 * pi)]:
    aa = np.unwrap(np.angle(1 - r * np.exp(-1j * tt)))
    assert abs((aa[-1] - aa[0]) - expected) < 1e-9

# ---------- Problem 30 (and Paper connection 6.7 spacings) ----------
P2 = 2 * pi / log2
g300 = 2 * pi / math.log(300 / (2 * pi))
g650 = 2 * pi / math.log(650 / (2 * pi))
assert abs(math.log(300 / (2 * pi)) - 3.8659) < 1e-4
assert abs(g300 - 1.6253) < 1e-4
assert abs(P2 / g300 - 5.58) < 0.01
assert abs(math.log(650 / (2 * pi)) - 4.6391) < 1e-4
assert abs(g650 - 1.3544) < 1e-4
assert abs(P2 / g650 - 6.69) < 0.01
assert abs(P2 / g650 - 6.7) < 0.05
t400 = float(get_zeros(400)[399])
assert abs(t400 - 680) < 5
ratio = math.log(t400 / (2 * pi)) / log2
assert abs(ratio - P2 / (2 * pi / math.log(t400 / (2 * pi)))) < 1e-12
assert abs(ratio - 6.8) < 0.06

print("All chapter checks passed.")
