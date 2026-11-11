import numpy as np, mpmath as mp, qmnt
from scipy.optimize import brentq
from scipy.special import lambertw

def Nsm(t):
    u = t / (2 * np.pi); return u * np.log(u) - u + 7 / 8
def s_lam(n):
    return 2 * np.pi * (n - 11 / 8) / lambertw((n - 11 / 8) / np.e).real
def gap(s):
    return 2 * np.pi / np.log(s / (2 * np.pi))
def th_asym(t):
    return t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8 + 1 / (48 * t)

t = np.asarray(qmnt.zeros(2000))
def N(T): return int(np.searchsorted(t, T, side="right"))

# Sample 1
assert (N(30), N(40), N(50)) == (3, 6, 10)
assert abs(float(mp.zetazero(11).imag) - 52.97) < 0.01
assert abs((t[9] - t[0]) / 9 - 3.9599) < 1e-3
# Problem 1
assert (N(20), N(35), N(48.1)) == (1, 5, 9)
# Sample 2 and Nsm values
assert abs(Nsm(50) - 9.422782) < 1e-5
assert abs(Nsm(30) - 3.564656) < 1e-5 and abs(Nsm(40) - 6.292650) < 1e-5
assert abs(Nsm(100) - 29.002344) < 1e-5 and N(100) == 29
assert abs(1 / (48 * np.pi * 14.134725) - 4.7e-4) < 1e-5
# Sample: winding
phi = np.linspace(0, 2 * np.pi, 20001); z = 2.5 * np.exp(1j * phi)
ph = np.unwrap(np.angle((z - 1) * (z - 2) * (z + 3)))
assert abs((ph[-1] - ph[0]) / (2 * np.pi) - 2) < 1e-9
# Problem 6
for R, c, ans in [(2, 0, 3), (0.5, 1, 1)]:
    zz = c + R * np.exp(1j * phi); p = np.unwrap(np.angle(zz**3 - 1))
    assert abs((p[-1] - p[0]) / (2 * np.pi) - ans) < 1e-9
# Theta sample at 50
assert abs(th_asym(50) - 26.461366) < 1e-5
assert abs(float(mp.siegeltheta(50)) - 26.461366) < 1e-5
S50 = 10 - 1 - float(mp.siegeltheta(50)) / np.pi
assert abs(S50 - 0.577086) < 1e-5
assert abs(float(mp.backlunds(50)) - S50) < 1e-6
# Sample S values
for T, Nv, cl in [(30, 3, -0.564877), (40, 6, -0.292816), (100, 29, -0.002410)]:
    S = Nv - 1 - float(mp.siegeltheta(T)) / np.pi
    assert abs(S - cl) < 2e-5, T
    assert abs(float(mp.backlunds(T)) - S) < 1e-6
# Backlund 79 zeros below 200
assert t[78] < 200 < t[79]
# loglog values
assert abs(np.log(np.log(1e3)) - 1.9) < 0.05 and abs(np.log(np.log(1e4)) - 2.2) < 0.05
# Newton s1 sample
assert abs(Nsm(14.134725) - 0.449278) < 1e-5
s1 = brentq(lambda x: Nsm(x) - 0.5, 2 * np.pi, 2 * np.pi * 11, xtol=1e-13)
assert abs(s1 - 14.521346) < 1e-5 and abs(s_lam(1) - s1) < 1e-8
assert abs(lambertw(-0.375 / np.e).real - (-0.16226)) < 1e-4
assert abs(t[0] - s1 - (-0.386621)) < 1e-5
assert abs(gap(s1) - 7.5001) < 1e-3
assert abs((t[0] - s1) / gap(s1) - (-0.0515)) < 1e-3
# Lambert s10 sample
assert abs(lambertw(8.625 / np.e).real - 1.078808) < 1e-5
s10 = s_lam(10)
assert abs(s10 - 50.2337) < 1e-3 and abs(gap(s10) - 3.0225) < 1e-3
assert abs((t[9] - s10) / gap(s10) - (-0.152)) < 1e-3
# Table
for T, ns, g in [(50, 9.423, 3.029), (100, 29.002, 2.271), (811.184, 499.296, 1.293),
                 (1000, 648.616, 1.239), (1419.422, 999.418, 1.159), (10000, 10142.965, 0.852)]:
    assert abs(Nsm(T) - ns) < 1.5e-3, T
    assert abs(gap(T) - g) < 1.5e-3, T
# faces: second order term, S(t1)
assert abs(0.386621**2 / (4 * np.pi * 14.521346) - 8.2e-4) < 1e-5
S_t1 = 0.5 - 1 - float(mp.siegeltheta(t[0])) / np.pi
assert abs(S_t1 - 0.0503) < 2e-4
# unfolding sample
assert abs(Nsm(t[9]) - 9.348203) < 1e-5
assert abs((Nsm(t[9]) - Nsm(t[0])) / 9 - 0.98877) < 1e-5
# nulls
for (a, b), cl in [((401, 500), 0.263), ((501, 1000), 0.270), ((1001, 2000), 0.279)]:
    n = np.arange(a, b + 1)
    assert abs(qmnt.rms(qmnt.fluct(t[n - 1], n)) - cl) < 6e-4
# np.gradient vs analytic at n=450
ss = np.array([s_lam(k) for k in range(1, 1001)])
gg = np.gradient(ss)
assert abs(gg[449] / gap(ss[449]) - 1) < 1e-4
# paper connection numbers
assert abs(Nsm(1419.42248094599) - 999.5 - (-0.082)) < 1e-3
assert abs(0.263 * gap(811.1843588465) - 0.34) < 0.01
assert abs(2 * gap(s_lam(450)) - 2.6) < 0.05
# Problem 2
assert abs(Nsm(2 * np.pi) + 0.125) < 1e-12 and abs(Nsm(2 * np.pi * np.e) - 0.875) < 1e-12
assert N(2 * np.pi * np.e) == 1
# Problems 4, 5
assert abs(gap(1000) - 1.239) < 1e-3 and abs(0.1 / gap(1000) - 0.0807) < 1e-4
# Problem 9
assert abs(t[99] - 236.524230) < 1e-5
assert abs(lambertw(98.625 / np.e).real - 2.625901) < 1e-5
s100 = s_lam(100)
assert abs(s100 - 235.987) < 1e-3 and abs(gap(s100) - 1.733) < 1e-3
assert abs((t[99] - s100) / gap(s100) - 0.310) < 1e-3
# Problem 13
assert abs(-np.log(30 / (2 * np.pi)) / (2 * np.pi) - (-0.249)) < 1e-3
# Problem 17
assert abs(Nsm(1001) - Nsm(1000) - 0.807) < 1e-3
# Problem 19
g0 = brentq(th_asym, 10, 30)
assert abs(g0 - 17.8456) < 1e-3 and abs(float(mp.grampoint(0)) - 17.8456) < 1e-3
assert abs(th_asym(17.8) - (-0.02411)) < 1e-4
# Problem 21
assert abs((Nsm(t[999]) - Nsm(t[0])) / 999 - 0.99997) < 2e-5
# Problem 23
mp.mp.dps = 30
for tv in [50, 200]:
    tt = mp.mpf(tv)
    r = tt**3 * (mp.siegeltheta(tt) - tt / 2 * mp.log(tt / (2 * mp.pi)) + tt / 2 + mp.pi / 8 - 1 / (48 * tt))
    assert abs(float(r) - 7 / 5760) < 2e-6
# Problem 27
sq = np.asarray(qmnt.smooth_zeros(np.arange(1, 1001)))
assert np.max(np.abs(sq - ss)) < 1e-6
print("all checks passed")
