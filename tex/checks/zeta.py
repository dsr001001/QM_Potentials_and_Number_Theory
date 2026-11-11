import numpy as np, mpmath
mpmath.mp.dps = 25

def close(a, b, tol):
    assert abs(float(a) - float(b)) < tol, (a, b)

# Table tab:zeta-values
close(mpmath.zeta(1.5), 2.612375, 1e-6)
close(mpmath.zeta(2), 1.644934, 1e-6)
close(mpmath.zeta(3), 1.202057, 1e-6)
close(mpmath.zeta(4), 1.082323, 1e-6)
close(np.pi**4 / 90, 1.082323, 1e-6)

# Sample: partial sums of zeta(2)
S10 = sum(1.0 / k**2 for k in range(1, 11))
close(S10, 1.549768, 1e-6)
close(S10 + 1 / 11, 1.640677, 1e-6)
close(S10 + 0.1, 1.649768, 1e-6)

# Checkpoint zeta(1.001)
z = mpmath.zeta(1.001)
assert 1000 <= z <= 1001

# Sample: two arrows at s=2+3i
s = 2 + 3j
a2 = 2.0**(-s); a3 = 3.0**(-s)
close(a2.real, -0.1217, 1e-4); close(a2.imag, -0.2184, 1e-4)
close(a3.real, -0.1098, 1e-4); close(a3.imag, 0.0171, 1e-4)
close(3 * np.log(2), 2.0794, 1e-4); close(3 * np.log(3), 3.2958, 1e-4)
assert abs(complex(mpmath.zeta(s))) <= np.pi**2 / 6

# Checkpoint zeta(3+100i) bounds
za = abs(complex(mpmath.zeta(3 + 100j)))
assert 1 / 1.202057 <= za <= 1.202057
close(1 / float(mpmath.zeta(3)), 0.831907, 1e-6)

# Table tab:zeta-euler
E = 1.0
expect = {2: 1.333333, 3: 1.5, 5: 1.5625, 7: 1.595052, 11: 1.608344, 13: 1.617918}
err = {2: 0.311601, 3: 0.144934, 5: 0.082434, 7: 0.049882, 11: 0.036590, 13: 0.027016}
for p in [2, 3, 5, 7, 11, 13]:
    E *= 1 / (1 - p**-2.0)
    close(E, expect[p], 1e-6)
    close(np.pi**2 / 6 - E, err[p], 1e-6)
    assert np.pi**2 / 6 - E <= 1 / p

# Sample: product beats harmonic sum x=10
close(210 / 48, 4.375, 1e-12)
close(sum(1 / k for k in range(1, 11)), 2.928968, 1e-6)

# Period 2pi/log2
close(2 * np.pi / np.log(2), 9.0647, 1e-4)
close(2 * np.pi / np.log(2), 9.064720, 1e-6)

# Sample: geometric at z=-2
for N, v in zip(range(5), [1, -1, 3, -5, 11]):
    assert (1 - (-2)**(N + 1)) // 3 == v

# Sample: close to the pole
close(mpmath.zeta(1.01), 100.578, 1e-3)
close(1 - 2**-0.01, 0.0069075, 1e-7)
close(0.01 * np.log(2), 0.0069315, 1e-7)

# Sample: zeta(1/2) via eta (Dirichlet eta = mpmath.altzeta)
def eta_avg(s, N):
    k = np.arange(1, N + 2)
    terms = (-1.0)**(k + 1) * np.exp(-s * np.log(k))
    S = np.cumsum(terms)
    return 0.5 * (S[N - 1] + S[N])
e = eta_avg(0.5, 1000)
close(mpmath.altzeta(0.5), 0.604899, 1e-6)
close((1 - mpmath.sqrt(2)) * mpmath.zeta(0.5), 0.604899, 1e-6)
assert abs(e - float(mpmath.altzeta(0.5))) < 1e-5
close(mpmath.zeta(0.5), -1.460355, 1e-6)
close(1 - np.sqrt(2), -0.414214, 1e-6)
close(1001**-0.5, 0.0316, 1e-4)

# Figure / Problem 20: eta at rho1 small
t1 = float(mpmath.zetazero(1).imag)
assert abs(eta_avg(0.5 + 1j * t1, 1000)) < 1e-3

# Functional equation numeric check at s=0.3+7i
sp = mpmath.mpc(0.3, 7.0)
lhs = mpmath.zeta(sp)
rhs = 2**sp * mpmath.pi**(sp - 1) * mpmath.sin(mpmath.pi * sp / 2) * mpmath.gamma(1 - sp) * mpmath.zeta(1 - sp)
assert abs(lhs - rhs) < 1e-15
xi = lambda s: s * (s - 1) / 2 * mpmath.pi**(-s / 2) * mpmath.gamma(s / 2) * mpmath.zeta(s)
assert abs(xi(sp) - xi(1 - sp)) < 1e-15

# zeta(-1), zeta(0), zeta(-3), trivial zeros
close(mpmath.zeta(-1), -1 / 12, 1e-12)
close(mpmath.zeta(0), -0.5, 1e-12)
close(mpmath.zeta(-3), 1 / 120, 1e-12)
for n in [-2, -4, -6, -8, -10]:
    assert abs(mpmath.zeta(n)) < 1e-20

# Table tab:zeta-zeros (RE1) and gaps
tz = [float(mpmath.zetazero(n).imag) for n in range(1, 12)]
RE1 = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
       37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
gaps = [6.887315, 3.988818, 5.414018, 2.510186, 4.651116,
        3.332541, 2.408354, 4.678078, 1.768681]
for a, b in zip(tz, RE1):
    close(a, b, 1e-6)
for i, g in enumerate(gaps):
    close(tz[i + 1] - tz[i], g, 2e-6)
assert tz[10] > 50
for n in range(1, 11):
    assert abs(mpmath.zetazero(n).real - 0.5) < 1e-20

# Bisection sample
a, b = 14.0, 15.0
Za = mpmath.siegelz(a); assert Za < 0
assert mpmath.siegelz(15) > 0
mids = [14.5, 14.25, 14.125, 14.1875, 14.15625, 14.140625, 14.1328125, 14.13671875]
signs = "++-+++-+"
for step in range(8):
    m = 0.5 * (a + b)
    assert m == mids[step]
    Zm = mpmath.siegelz(m)
    assert (Zm > 0) == (signs[step] == "+")
    if Zm * Za < 0: b = m
    else: a, Za = m, Zm
assert (a, b) == (14.1328125, 14.13671875)
est = 0.5 * (a + b)
close(est, 14.134765625, 1e-12)
close(abs(est - t1), 4.0e-5, 1e-6)
close(1 / 2**8, 0.00390625, 1e-12)
close(1 / 2**9, 0.001953, 1e-6)
close(2**-20, 9.5e-7, 1e-8)
close(mpmath.siegelz(0), -1.460355, 1e-6)
# Checkpoint: Z(30)>0, Z(31)<0
assert mpmath.siegelz(30) > 0 and mpmath.siegelz(31) < 0
# Z real
assert abs(mpmath.im(mpmath.exp(1j * mpmath.siegeltheta(20)) * mpmath.zeta(0.5 + 20j))) < 1e-15

# Paper anchors t1, t500, t1000 (mpmath)
mpmath.mp.dps = 20
close(mpmath.zetazero(1).imag, 14.134725141734695, 1e-14)
close(mpmath.zetazero(500).imag, 811.1843588465, 1e-9)
t1000 = float(mpmath.zetazero(1000).imag)
close(t1000, 1419.42248094599, 1e-10)
mpmath.mp.dps = 25

# Problem 21: mean spacing at t1000
g = 2 * np.pi / np.log(t1000 / (2 * np.pi))
close(g, 1.159, 1e-3)
close(t1000 / (2 * np.pi), 225.91, 1e-2)
close(np.log(t1000 / (2 * np.pi)), 5.4201, 1e-3)

# Problem 1
S5 = sum(1.0 / k**2 for k in range(1, 6))
close(S5, 1.463611, 1e-6)
close(S5 + 1 / 6, 1.630278, 1e-6)
close(S5 + 0.2, 1.663611, 1e-6)
# Problem 2, 3
close(4**-1.5, 0.125, 1e-12)
close(5**-0.5, 0.447214, 1e-6); close(np.log(5), 1.609438, 1e-6)
# Problem 9
prod20 = 1.0
for p in [2, 3, 5, 7, 11, 13, 17, 19]:
    prod20 *= p / (p - 1)
close(prod20, 5.847132, 1e-6)
close(sum(1 / k for k in range(1, 21)), 3.597740, 1e-6)
# Problem 10
close(mpmath.zeta(1.1), 10.584, 1e-3)
# Problem 11
close(np.pi**2 / 12, 0.822467, 1e-6)
close(mpmath.altzeta(2), np.pi**2 / 12, 1e-12)
# Problem 23
close(2**-0.5, 0.707107, 1e-6); close(np.log(2), 0.693147, 1e-6)
# Problem 27
close(float(mpmath.zeta(1.5)) - 2, 0.612375, 1e-6)
# Problem 25 identity numeric
s = 2.5 + 1j
c = sum((1 if k % 3 else -2) * k**(-s) for k in range(1, 200001))
assert abs(c - (1 - 3**(1 - s)) * complex(mpmath.zeta(s))) < 1e-6
print("all checks passed")
