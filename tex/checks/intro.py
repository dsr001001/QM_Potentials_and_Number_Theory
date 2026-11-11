import numpy as np
from scipy.optimize import brentq
import qmnt

twopi = 2 * np.pi
def gap(t):
    return twopi / np.log(t / twopi)
def Nsm(t):
    u = t / twopi
    return u * np.log(u) - u + 7.0 / 8.0

# RE1 values, Table and Sample Problem 1
t10 = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
tz = np.asarray(qmnt.zeros(500))
assert np.allclose(tz[:10], t10, atol=1e-6)
d = np.diff(t10)
claimed = [6.887315, 3.988818, 5.414018, 2.510186, 4.651116,
           3.332541, 2.408354, 4.678078, 1.768681]
assert np.allclose(d, claimed, atol=1e-6)
assert abs(t10[-1] - t10[0] - 35.639107) < 1e-6
assert abs(d.mean() - 3.959901) < 1e-6
assert abs(d.max() - 6.887315) < 1e-6 and abs(d.min() - 1.768681) < 1e-6

# pycode listing outputs
assert np.sum(np.asarray(qmnt.zeros(40)) <= 100.0) == 29
assert abs(float(qmnt.Nsm(100.0)) - 29.002350) < 1e-5

# Sample Problem 2: counting
assert abs(tz[10] - 52.970321) < 1e-5
assert abs(tz[28] - 98.831194) < 1e-5
assert abs(tz[29] - 101.317851) < 1e-5
assert np.sum(tz <= 50) == 10 and np.sum(tz <= 100) == 29
assert abs(50 / twopi - 7.957747) < 1e-6
assert abs(np.log(50 / twopi) - 2.074146) < 1e-6
assert abs(Nsm(50.0) - 9.422781) < 1e-5
assert abs(Nsm(100.0) - 29.002350) < 1e-5
assert abs((10 - Nsm(50.0)) - 0.577) < 1e-3
assert abs((29 - Nsm(100.0)) - (-0.002)) < 1e-3

# Staircase figure caption: departure below one step on 8 <= T <= 100
zs = tz[(tz >= 8) & (tz <= 100)]
dev = [abs(0 - Nsm(8.0)), abs(29 - Nsm(100.0))]
for k, z in enumerate(zs):
    nbelow = np.sum(tz < z)
    dev.append(abs(nbelow - Nsm(z)))        # just before the step
    dev.append(abs(nbelow + 1 - Nsm(z)))    # just after the step
assert max(dev) < 1.0

# Sample Problem 3: gap near zero 500
t500 = 811.1843588465
assert abs(tz[499] - t500) < 1e-6
assert abs(t500 / twopi - 129.10399) < 1e-4
assert abs(np.log(t500 / twopi) - 4.860618) < 1e-5
g500 = gap(t500)
assert abs(g500 - 1.292672) < 1e-5
assert abs(0.5 / g500 - 0.3868) < 1e-4
assert abs(3.959901 / g500 - 3.063) < 1e-3

# Sample Problem 4 (new): smooth zeros and fluctuations of zeros 1 and 2
f = lambda u: u * np.log(u) - u
assert abs(np.log(2.30) - 0.832909) < 1e-6
assert abs(np.log(2.32) - 0.841567) < 1e-6
assert abs(np.log(2.311) - 0.837680) < 1e-6
assert abs(f(2.30) - (-0.3843)) < 1e-4
assert abs(f(2.32) - (-0.3676)) < 1e-4
assert abs(f(2.311) - (-0.3751)) < 1e-4
assert abs((-0.375 - f(2.311)) - 0.00012) < 1e-5
assert abs(0.00012 / 0.838 - 0.00015) < 1e-5
u1 = brentq(lambda u: f(u) + 0.375, 2.0, 3.0)
assert abs(u1 - 2.31115) < 1e-5
s1 = twopi * u1
assert abs(s1 - 14.5214) < 2e-4
assert abs(np.log(u1) - 0.83775) < 1e-5
assert abs(gap(s1) - 7.500) < 1e-3
fl1 = (tz[0] - s1) / gap(s1)
assert abs((tz[0] - s1) - (-0.3867)) < 2e-4
assert abs(fl1 - (-0.0516)) < 3e-4 and abs(fl1 - (-0.052)) < 1e-3
assert abs(np.log(3.28) - 1.187843) < 1e-6
assert abs(np.log(3.29) - 1.190887) < 1e-6
assert abs(f(3.28) - 0.6161) < 1e-4
assert abs(f(3.29) - 0.6280) < 1e-4
assert abs(f(3.2875) - 0.6250) < 1e-4
u2 = brentq(lambda u: f(u) - 0.625, 3.0, 4.0)
assert abs(u2 - 3.28746) < 1e-5
s2 = twopi * u2
assert abs(s2 - 20.6557) < 2e-4
assert abs(np.log(u2) - 1.19012) < 1e-5
assert abs(gap(s2) - 5.2795) < 2e-4
fl2 = (tz[1] - s2) / gap(s2)
assert abs((tz[1] - s2) - 0.3663) < 2e-4
assert abs(fl2 - 0.0694) < 3e-4 and abs(fl2 - 0.069) < 1e-3
sq = np.asarray(qmnt.smooth_zeros(np.array([1, 2])))
assert np.allclose(sq, [s1, s2], atol=1e-3)
fq = np.asarray(qmnt.fluct(tz[:2], np.array([1, 2])))
assert np.allclose(fq, [fl1, fl2], atol=1e-3)

# Sample Problem 5 (prime period) and Problems 3, 11, 20, 24
P = twopi / np.log(2)
assert abs(P - 9.064720) < 1e-6
assert abs(P / 2 - 4.532) < 1e-3 and abs(P / 3 - 3.022) < 1e-3
g50 = gap(50.0)
assert abs(g50 - 3.029288) < 1e-5
assert abs(P / g500 - 7.012) < 1e-3
assert abs(P / g50 - 2.992) < 1e-3
assert abs(0.6 / g50 - 0.198) < 1e-3
assert abs(0.5 / g50 - 0.165) < 1e-3
assert abs(0.5 / g500 - 0.387) < 1e-3

# Sample Problem 6: ruler
x = np.array([0, 1, 2, 3, 4], float)
y = np.array([1.0, 2.9, 5.2, 6.9, 9.1])
slope = (y[-1] - y[0]) / 4
assert abs(slope - 2.025) < 1e-12
model = 1.0 + slope * x
res = y - model
assert np.allclose(res, [0, -0.125, 0.15, -0.175, 0], atol=1e-12)
r = np.sqrt(np.mean(res**2))
assert abs(np.sum(res**2) - 0.06875) < 1e-12
assert abs(r - 0.11726) < 1e-5
assert abs((1.0 + slope * 5) - 11.125) < 1e-12

# Sample Problem 7 and Problem 23: memorizing polynomial and difference table
assert np.allclose(np.diff(y, 1), [1.9, 2.3, 1.7, 2.2], atol=1e-12)
assert np.allclose(np.diff(y, 2), [0.4, -0.6, 0.5], atol=1e-12)
assert np.allclose(np.diff(y, 3), [-1.0, 1.1], atol=1e-12)
assert abs(np.diff(y, 4)[0] - 2.1) < 1e-12
d3 = 1.1 + 2.1; d2 = 0.5 + d3; d1 = 2.2 + d2; y5 = 9.1 + d1
assert abs(d3 - 3.2) < 1e-12 and abs(d2 - 3.7) < 1e-12 and abs(d1 - 5.9) < 1e-12
assert abs(y5 - 15.0) < 1e-12
e3 = d3 + 2.1; e2 = d2 + e3; e1 = d1 + e2; y6 = y5 + e1
assert abs(e3 - 5.3) < 1e-12 and abs(e2 - 9.0) < 1e-12 and abs(e1 - 14.9) < 1e-12
assert abs(y6 - 29.9) < 1e-12
p4 = np.polyfit(x, y, 4)
assert np.allclose(np.polyval(p4, x), y, atol=1e-9)
assert abs(np.polyval(p4, 5.0) - 15.0) < 1e-8
assert abs(np.polyval(p4, 6.0) - 29.9) < 1e-7
assert abs((15.0 - 11.0) / 0.125 - 32) < 1e-12
assert abs((1.0 + slope * 6) - 13.15) < 1e-12

# Sample Problem 8 and Problem 8: exams
assert abs((4 + 16 * 0.25) / 20 - 0.40) < 1e-12
assert abs(0.85 * 20 - 17) < 1e-12
assert abs(20 * 0.25 - 5) < 1e-12
assert abs((5 + 20 * 0.2) / 25 - 0.36) < 1e-12
assert abs(25 * 0.2 / 25 - 0.20) < 1e-12

# Sample Problem 9 (rms and correlation)
truth = np.array([0.2, -0.3, 0.1, 0.4, -0.4])
rnull = np.sqrt(np.mean(truth**2))
assert abs(np.sum(truth**2) - 0.46) < 1e-12
assert abs(rnull - 0.3033) < 1e-4
assert abs(np.sqrt(np.mean((truth - 0.5 * truth)**2)) - 0.1517) < 1e-4
assert abs(np.sqrt(np.mean((truth + 0.5 * truth)**2)) - 0.4550) < 1e-4
assert abs(np.corrcoef(truth, 0.5 * truth)[0, 1] - 1) < 1e-12
assert abs(np.corrcoef(truth, -0.5 * truth)[0, 1] + 1) < 1e-12

# Sample Problem 10 and Problems 6, 7, 9, 16, 17 (paper-number ratios)
assert abs(0.369 / 0.263 - 1.403) < 1e-3
assert abs(0.074 / 0.263 - 0.281) < 1e-3
assert abs((1 - 0.074 / 0.263) - 0.719) < 1e-3
assert abs((1 - 0.043 / 0.263) - 0.837) < 1e-3
assert abs((1 - 0.034 / 0.263) - 0.871) < 1e-3
assert abs(0.265 / 0.263 - 1.0076) < 1e-4
assert abs(900 / 500 - 1.8) < 1e-12
assert len(range(370, 401)) == 31 and len(range(401, 461)) == 60
assert abs(2.64 / 0.255 - 10.35) < 0.01

# Problem 5
y5p = np.array([2.0, 3.4, 5.1, 6.4])
s5 = (y5p[-1] - y5p[0]) / 3
assert abs(s5 - 1.4667) < 1e-4
assert abs((3.4 - (2.0 + s5)) - (-0.0667)) < 1e-4
assert abs((5.1 - (2.0 + 2 * s5)) - 0.1667) < 1e-4
assert abs((2.0 + 4 * s5) - 7.867) < 1e-3

# Problem 10
e10 = np.array([0.3, -0.1, 0.0, -0.2])
assert abs(np.sqrt(np.mean(e10**2)) - 0.187) < 1e-3

# Problem 12
R = 0.263
for a, val in [(0, 0.263), (0.5, 0.1315), (2, 0.263), (-1, 0.526)]:
    tr = R * truth / rnull
    assert abs(np.sqrt(np.mean((tr - a * tr)**2)) - val) < 1e-9

# Problem 13
t100 = tz[99]
assert abs(t100 - 236.524230) < 1e-5
assert abs((t100 - tz[0]) / 99 - 2.246359) < 1e-5

# Problem 15: null and six-prime oracle on zeros 401-500 (paper Table V)
n = np.arange(401, 501)
null = qmnt.rms(qmnt.fluct(tz[n - 1], n))
assert abs(null - 0.2631) < 0.002
pred = qmnt.oracle_zeros(n, qmnt.first_primes(6))
rr, cc = qmnt.score(pred, n)
assert abs(rr - 0.074) < 0.003
assert abs(cc - 0.96) < 0.01

print('all intro checks passed')
