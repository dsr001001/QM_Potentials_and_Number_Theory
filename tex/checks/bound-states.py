import numpy as np, mpmath, qmnt
from scipy.linalg import eigh_tridiagonal
from scipy.special import ai_zeros, erfc, beta
from scipy.integrate import quad

pi = np.pi

def levels(Vfun, L, N, k):
    h = 2*L/(N + 1)
    x = -L +h*np.arange(1, N + 1)
    return eigh_tridiagonal(2/h**2 + Vfun(x), -np.ones(N - 1)/h**2,
                            eigvals_only=True, select="i",
                            select_range=(0, k - 1))

# Sample "Empty box of the paper" (RE3)
L1 = 44.883
E1 = pi**2/(4*L1**2)
assert abs(E1 - 1.2248e-3) < 1e-7
assert abs(500**2*E1 - 306.2) < 0.1
assert int(np.floor(2*L1*np.sqrt(1000)/pi)) == 903
assert abs(2*L1*np.sqrt(1000)/pi - 903.6) < 0.05
assert abs(1001*E1 - 1.226) < 1e-3

# Sample "Where is the particle"
assert abs(0.5 + 1/pi - 0.818) < 1e-3

# Sample "Checking an oscillator state"
assert abs(1/np.sqrt(2) - 0.7071) < 1e-4 and abs(np.sqrt(5) - 2.2361) < 1e-4
val = quad(lambda x: (2*x**2 - 1)**2*np.exp(-x**2), -np.inf, np.inf)[0]
assert abs(val - 2*np.sqrt(pi)) < 1e-8
assert abs(quad(lambda x: np.exp(-x**2), -np.inf, np.inf)[0] - 1.7724539) < 1e-6

# Listing: oscillator levels, error below 0.01 and growing with n
E = levels(lambda x: x**2, 8.0, 2000, 10)
err = E - (2*np.arange(1, 11) - 1)
assert np.all(np.abs(err) < 0.01)
assert np.all(np.diff(np.abs(err)) > 0)

# Sample "Distant wall": growing solution g = exp(x^2/2) solves with E=-1
xg, dg = 3.0, 1e-4
g = lambda x: np.exp(x**2/2)
gpp = (g(xg + dg) - 2*g(xg) + g(xg - dg))/dg**2
assert abs((-gpp + xg**2*g(xg))/g(xg) - (-1)) < 1e-3
assert abs(8/np.sqrt(pi)*np.exp(-4) - 0.083) < 1e-3
Eb2 = levels(lambda x: x**2, 2.0, 6000, 1)[0]
assert abs(Eb2 - 1.075) < 0.002
assert abs((Eb2 - 1) - 0.075) < 0.002
ratio = (8/np.sqrt(pi)*np.exp(-4))/(Eb2 - 1)
assert 1.05 < ratio < 1.15   # "about 10% high"
assert abs(16/np.sqrt(pi)*np.exp(-16) - 1.0e-6) < 0.05e-6
Eb4 = levels(lambda x: x**2, 4.0, 8000, 1)[0]
Eref = levels(lambda x: x**2, 8.0, 16000, 1)[0]
assert abs(Eb4 - Eref) < 1e-5

# Table and V-shaped sample
a, ap, _, _ = ai_zeros(8)
lin = np.sort(np.concatenate([-a, -ap]))
for got, want in zip(lin[:6], [1.0188, 2.3381, 3.2482, 4.0879, 4.8201, 5.5206]):
    assert abs(got - want) < 1e-4
cnt = lambda n: (3*pi/4*(n - 0.5))**(2/3)
for n, w in [(1, 1.115), (2, 2.320), (9, 7.375), (10, 7.942)]:
    assert abs(cnt(n) - w) < 1e-3
assert abs(lin[8] - 7.372) < 1e-3 and abs(lin[9] - 7.944) < 1e-3
assert 0.08 < cnt(1)/lin[0] - 1 < 0.12           # about 10% off
assert abs(cnt(9) - lin[8]) < 0.0035 and abs(cnt(10) - lin[9]) < 0.0035

# Leibniz check: J_alpha = (1 + 2/alpha) I_alpha
I4 = quad(lambda u: np.sqrt(1 - u**4), 0, 1)[0]
J4 = beta(0.25, 0.5)/4
assert abs(J4 - 1.3110) < 1e-4 and abs(1.5*I4 - 1.3110) < 1e-4
J2 = quad(lambda u: 1/np.sqrt(1 - u**2), 0, 1)[0]
assert abs(J2 - pi/2) < 1e-6

# Density of zeros, sample "What shape"
rz = lambda t: np.log(t/(2*pi))/(2*pi)
assert abs(rz(14.134725) - 0.129) < 1e-3
assert abs(rz(1419.42248) - 0.863) < 1e-3
t500 = 811.1843588465
assert abs(t500/(2*pi) - 129.10) < 0.01
lg = np.log(t500/(2*pi)); assert abs(lg - 4.8606) < 1e-4
assert abs(1/lg - 0.2057) < 1e-4
alpha = 1/(0.5 + 1/lg); assert abs(alpha - 1.417) < 1e-3
assert abs(rz(t500) - 0.774) < 1e-3 and abs(1/rz(t500) - 1.293) < 1e-3
rb = L1/(pi*np.sqrt(t500)); assert abs(rb - 0.502) < 1e-3 and abs(1/rb - 1.99) < 0.01
assert abs(np.sqrt(t500) - 28.481) < 1e-3
assert abs(rz(t500)/rb - 1.54) < 0.01   # Problem 21

# Paper connection
assert abs(qmnt.L_of(1000) - 44.883) < 1e-3
assert abs(qmnt.V0 - 9.741) < 1e-3
xt = qmnt.wu_sprung_x(t500); assert abs(xt - 38.5) < 0.1
assert abs(L1 - xt - 6.4) < 0.1
assert abs(1000 - t500 - 190) < 2
Nsm1000 = float(qmnt.Nsm(1000.0)); assert abs(Nsm1000 - 648.6) < 0.05
# turning-point band of levels 201--300 (paper Sec. III: 22.5--28.5)
tz = qmnt.zeros(300)
x201 = qmnt.wu_sprung_x(tz[200]); x300 = qmnt.wu_sprung_x(tz[299])
assert abs(x201 - 22.5) < 0.5 and abs(x300 - 28.5) < 0.5

# Problem 1
for n, w in [(1, 2.467), (2, 9.870), (3, 22.21), (4, 39.48)]:
    assert abs(n**2*pi**2/4 - w) < 0.01
for m, w in [(3, 7.402), (5, 12.34), (7, 17.27)]:
    assert abs(m*pi**2/4 - w) < 0.01
# Problem 5
f = lambda x: x**2*np.cos(pi*x/2)**2
assert abs(quad(f, -1, 1)[0] - 0.1307) < 1e-4
assert abs(1/3 - 2/pi**2 - 0.1307) < 1e-4
# Problem 13
assert abs(erfc(1) - 0.1573) < 1e-4
# Problem 15
assert abs(4*1.0188 - 4.075) < 1e-3
# Problem 17
assert abs(I4 - 0.8740) < 1e-4
est = (pi*9.5/(2*I4))**(4/3); assert abs(pi*9.5/(2*I4) - 17.07) < 0.01
assert abs(est - 43.96) < 0.05
E10 = levels(lambda x: x**4, 6.0, 4000, 10)[9]; assert abs(E10 - 43.98) < 0.03
# Problem 23
for V, w in [(1000, 44.86), (1500, 59.93)]:
    assert abs(np.sqrt(V)/pi*np.log(2*V/(pi*np.e**2)) - w) < 0.01
assert abs(np.log(2000/pi) - 6.4562) < 1e-4 and abs(np.log(3000/pi) - 6.8616) < 1e-4
assert abs(qmnt.L_of(1500) - 59.957) < 1e-3
# Problem 25
assert abs(12/np.sqrt(pi)*np.exp(-9) - 8.4e-4) < 0.05e-4
Eb3 = levels(lambda x: x**2, 3.0, 8000, 1)[0]
Eref3 = levels(lambda x: x**2, 8.0, 21000, 1)[0]
assert abs((Eb3 - Eref3) - 7.9e-4) < 0.3e-4
# Problem 31
for n, w in [(1, 0.818), (3, 0.394), (5, 0.564)]:
    assert abs(0.5 + np.sin(n*pi/2)/(n*pi) - w) < 1e-3
print("all checks passed")
