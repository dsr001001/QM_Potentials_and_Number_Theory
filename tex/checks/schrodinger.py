import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad
from scipy.optimize import brentq
import mpmath
try:
    import qmnt
    HAVE_QMNT = True
except Exception:
    HAVE_QMNT = False

h_P = 6.62607015e-34
hbar = h_P/(2*np.pi)
me = 9.1093837e-31
mn = 1.6749275e-27
eV = 1.602176634e-19
c = 299792458.0

# hbar value quoted
assert abs(hbar - 1.054571817e-34) < 1e-42

# Sample: de Broglie, electron 100 eV and baseball
p = np.sqrt(2*me*100*eV)
assert abs(p - 5.403e-24) < 0.002e-24
lam = h_P/p
assert abs(lam*1e9 - 0.1226) < 0.0001
lam_ball = h_P/(0.145*40)
assert abs(lam_ball - 1.14e-34) < 0.005e-34

# Sample: free electron k = 1e10
k = 1.0e10
K = hbar**2*k**2/(2*me)
assert abs(K - 6.104e-19) < 0.002e-19
assert abs(K/eV - 3.81) < 0.005
om = K/hbar
assert abs(om - 5.79e15) < 0.005e15
assert abs(om/k - 5.79e5) < 0.005e5
assert abs(hbar*k/me - 1.16e6) < 0.005e6

# Sample: phase period for 1 eV
assert abs(h_P/eV - 4.14e-15) < 0.005e-15

# Sample: cusp normalization, P(|x|<a) = 1 - e^-2
a = 1.0
norm = quad(lambda x: np.exp(-2*abs(x)/a), -50, 50, points=[0])[0]
assert abs(norm - a) < 1e-8
P = quad(lambda x: np.exp(-2*abs(x)/a)/a, -a, a, points=[0])[0]
assert abs(P - 0.8647) < 1e-4
assert abs(1 - np.exp(-2) - 0.8647) < 1e-4

# Sample: box ground state, middle half = 1/2 + 1/pi
L = 1.0
psi1 = lambda x: np.sin(np.pi*(x + L)/(2*L))/np.sqrt(L)
assert abs(quad(lambda x: psi1(x)**2, -L, L)[0] - 1) < 1e-10
Pm = quad(lambda x: psi1(x)**2, -L/2, L/2)[0]
assert abs(Pm - (0.5 + 1/np.pi)) < 1e-10
assert abs(Pm - 0.8183) < 1e-4

# normalization 1/sqrt(L) for k = 1..5
for kk in range(1, 6):
    val = quad(lambda x: (np.sin(kk*np.pi*(x + L)/(2*L))**2)/L, -L, L, limit=200)[0]
    assert abs(val - 1) < 1e-8

# Sample RE3: E1 for L = 44.883 and <H> = 7 E1/4
Lp = 44.883
E1 = (np.pi/(2*Lp))**2
assert abs(np.pi/(2*Lp) - 0.0349976) < 1e-7
assert abs(E1 - 1.2248e-3) < 0.0001e-3
assert abs(4*E1 - 4.8993e-3) < 0.0002e-3
assert abs(7*E1/4 - 2.143e-3) < 0.001e-3

# Sample units: electron, l = 1 nm
eps = hbar**2/(2*me*(1e-9)**2)
assert abs(eps - 6.104e-21) < 0.002e-21
assert abs(eps/eV - 0.0381) < 0.00005
assert abs(np.pi**2 - 9.870) < 0.0005
assert abs(np.pi**2*eps/eV - 0.376) < 0.0005
# check against h^2/(8 m a^2) for a = 1 nm
assert abs(h_P**2/(8*me*(1e-9)**2)/eV - np.pi**2*eps/eV) < 1e-10

# Sample: phase pi rotation time at E = t1
t1 = 14.134725
assert abs(np.pi/t1 - 0.2223) < 0.00005

# Spectrum: Balmer line
dE = 13.6*(1/4 - 1/9)
assert abs(dE - 1.89) < 0.005
hc_eVnm = h_P*c/eV*1e9
assert abs(hc_eVnm - 1240) < 0.5
assert abs(hc_eVnm/dE - 656) < 1.5

# Sample: counting the paper's box
cnt = 2*Lp*np.sqrt(1000)/np.pi
assert abs(cnt - 903.57) < 0.01
assert int(np.floor(cnt)) == 903
assert abs(np.pi*np.sqrt(1000)/Lp - 2.213) < 0.001

# Nsm values and zero counts
def Nsm(t):
    return t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
assert abs(Nsm(100.0) - 29.0) < 0.05
assert abs(Nsm(1000.0) - 648.6) < 0.05
assert abs(2*np.pi/np.log(1000/(2*np.pi)) - 1.239) < 0.0005

def zeros_first(n):
    if HAVE_QMNT:
        return np.asarray(qmnt.zeros(n), dtype=float)
    return np.array([float(mpmath.zetazero(j).imag) for j in range(1, n + 1)])
if HAVE_QMNT:
    z = zeros_first(650)
    assert np.sum(z <= 100) == 29          # N(100) = 29, Problem 24
    assert np.sum(z <= 1000) == 649        # N(1000) = 649
else:
    z30 = zeros_first(30)
    assert np.sum(z30 <= 100) == 29
    t649 = float(mpmath.zetazero(649).imag); t650 = float(mpmath.zetazero(650).imag)
    assert t649 <= 1000 < t650
assert abs(zeros_first(1)[0] - 14.134725) < 1e-6

# Checkpoint: L for 649 levels about 32
assert abs(649*np.pi/(2*np.sqrt(1000)) - 32) < 0.5

# V0 value
V0 = 3.10073*np.pi
assert abs(V0 - 9.741) < 0.0005

# Wu-Sprung x(V): L values 44.883 and 59.957
def x_of_V(V):
    aa = np.sqrt(max(V - V0, 0.0)); bb = np.sqrt(V)
    return (aa*np.log(V0/(2*np.pi*np.e**2)) + bb*np.log((bb + aa)/(bb - aa)))/np.pi
assert abs(x_of_V(1000.0) - 44.883) < 0.001
assert abs(x_of_V(1500.0) - 59.957) < 0.001

# Problem 23 / Fig: first five WS levels within 0.1 mean spacings of s_n
def V_of_x(x):
    x = abs(x)
    if x == 0.0:
        return V0
    return brentq(lambda V: x_of_V(V) - x, V0, 5000.0, xtol=1e-12)
N = 6000
hh = 2*Lp/(N + 1)
xg = -Lp + hh*np.arange(1, N + 1)
half = np.array([V_of_x(xi) for xi in xg[N//2:]])
Vg = np.concatenate([half[::-1], half])
Ews = eigh_tridiagonal(2/hh**2 + Vg, -np.ones(N - 1)/hh**2, eigvals_only=True,
                       select='i', select_range=(0, 4))
s = np.array([brentq(lambda t: Nsm(t) - (n - 0.5), 2*np.pi, 1e4) for n in range(1, 6)])
gap = 2*np.pi/np.log(s/(2*np.pi))
assert np.all(np.abs((Ews - s)/gap) < 0.1)

# Paper-connection Sample and Problem 18: A_nk for the empty box
N2 = 4000
h2 = 2*Lp/(N2 + 1)
x2 = -Lp + h2*np.arange(1, N2 + 1)
E0, P0 = eigh_tridiagonal(2/h2**2*np.ones(N2), -np.ones(N2 - 1)/h2**2,
                          select='i', select_range=(0, 4))
P0 = P0/np.sqrt(np.sum(P0**2, axis=0)*h2)
for n in range(1, 6):
    for kk in range(0, 8):
        Ank = np.sum(P0[:, n - 1]**2*np.cos(kk*np.pi*x2/Lp))*h2
        if kk == 0:
            target = 1.0
        elif kk == n:
            target = (-1)**(n + 1)/2
        else:
            target = 0.0
        assert abs(Ank - target) < 1e-3
# orthogonality of numerical eigenvectors
assert abs(np.sum(P0[:, 0]*P0[:, 1])*h2) < 1e-8

# Problem 1: electron 10 eV
lam10 = h_P/np.sqrt(2*me*10*eV)
assert abs(lam10*1e9 - 0.388) < 0.0005
assert abs(np.sqrt(2*me*10*eV) - 1.708e-24) < 0.001e-24
assert abs(0.1226*np.sqrt(10) - 0.388) < 0.0005

# Problem 2: neutron 2200 m/s
assert abs(mn*2200 - 3.685e-24) < 0.001e-24
assert abs(h_P/(mn*2200)*1e9 - 0.180) < 0.0005

# Problem 5
assert abs(np.sqrt(25) - 5) < 1e-12
assert abs(2*np.pi/5 - 1.257) < 0.0005

# Problem 7: parabola trial function
L = 1.0
Ipar = quad(lambda x: (L**2 - x**2)**2, -L, L)[0]
assert abs(Ipar - 16/15*L**5) < 1e-12
A2 = 1/Ipar
EH = quad(lambda x: A2*(L**2 - x**2)*2.0, -L, L)[0]
assert abs(EH - 2.5/L**2) < 1e-10
assert abs(np.pi**2/4 - 2.467) < 0.0005
assert abs(EH/(np.pi**2/4) - 1.013) < 0.0005
assert abs(10/np.pi**2 - 1.013) < 0.0005

# Problem 8: <x^2> = L^2 (1/3 - 2/pi^2)
x2avg = quad(lambda x: x**2*psi1(x)**2, -L, L)[0]
assert abs(x2avg - L**2*(1/3 - 2/np.pi**2)) < 1e-10
assert abs(quad(lambda x: x*psi1(x)**2, -L, L)[0]) < 1e-12

# Problem 9
P9 = quad(lambda x: psi1(x)**2, L/2, L)[0]
assert abs(P9 - (0.25 - 1/(2*np.pi))) < 1e-10
assert abs(P9 - 0.0908) < 0.0001

# Problem 10 / Fig beats: |I| = 32 L/(9 pi^2) = 0.360 L, period with L = 44.883
psi2 = lambda x: np.sin(2*np.pi*(x + L)/(2*L))/np.sqrt(L)
I = quad(lambda x: x*psi1(x)*psi2(x), -L, L)[0]
assert abs(abs(I) - 32/(9*np.pi**2)) < 1e-10
assert abs(32/(9*np.pi**2) - 0.360) < 0.0005
period = 2*np.pi/(3*E1)
assert abs(period - 2*np.pi/(3*(np.pi/(2*44.883))**2)) < 1e-9

# Problem 11: half-life
assert abs(np.log(2)/0.1 - 6.93) < 0.005

# Problem 13: orthogonality
assert abs(quad(lambda x: psi1(x)*psi2(x), -L, L)[0]) < 1e-12

# Problem 14: electron in 0.2 nm box (L = 0.1 in units of 1 nm)
E14 = (np.pi/(2*0.1))**2*eps/eV
assert abs(E14 - 9.40) < 0.005

# Problem 16: E1 at both scales and ratio
E1s2 = (np.pi/(2*59.957))**2
assert abs(E1s2 - 6.864e-4) < 0.001e-4
assert abs(E1s2/E1 - 0.5604) < 0.0001
assert abs((44.883/59.957)**2 - 0.5604) < 0.0001

# Problem 21: FD relative error for k = 5, L = 1, N = 500
L = 1.0; N3 = 500
h3 = 2*L/(N3 + 1)
E3 = eigh_tridiagonal(2/h3**2*np.ones(N3), -np.ones(N3 - 1)/h3**2, eigvals_only=True,
                      select='i', select_range=(0, 4))
ex = (np.arange(1, 6)*np.pi/(2*L))**2
rel = 1 - E3/ex
assert np.all(rel > 0) and np.all(np.diff(rel) > 0)
assert abs(rel[4] - 8.19e-5) < 0.01e-5
aa = 5*np.pi*h3/(4*L)
assert abs(aa - 5*np.pi/1002) < 1e-15
assert abs((1 - np.sin(aa)**2/aa**2) - rel[4]) < 1e-9
disc = 4/h3**2*np.sin(np.arange(1, 6)*np.pi*h3/(4*L))**2
assert np.allclose(disc, E3, rtol=1e-10)

# Problem 22: numerical normalization of e^-|x|
hx = 0.001
xx = np.arange(-20, 20 + hx/2, hx)
ps = np.exp(-np.abs(xx))
ps = ps/np.sqrt(np.sum(ps**2)*hx)
P22 = np.sum(ps[np.abs(xx) < 1]**2)*hx
assert abs(P22 - 0.8647) < 0.002

# Problem 25: time unit
assert abs(hbar/eps - 1.73e-14) < 0.005e-14

# Problem 26: <p^2> = (pi/2L)^2 hbar^2 for box ground state (units hbar=1)
L = 1.0
d2 = lambda x: -(np.pi/(2*L))**2*psi1(x)
p2 = quad(lambda x: psi1(x)*(-d2(x)), -L, L)[0]
assert abs(p2 - (np.pi/(2*L))**2) < 1e-10

# Problem 27
assert abs(1/5 + 4/5 - 1) < 1e-15
assert abs((1/5)*1 + (4/5)*9 - 7.4) < 1e-12

# Problem 28 instance: envelope speed (k2^2-k1^2)/(k2-k1) = k1+k2 = 2k
k1, k2 = 1.7, 2.3
assert abs((k2**2 - k1**2)/(k2 - k1) - 2*(k1 + k2)/2) < 1e-12

# Checkpoint E = 100
assert abs(2*np.pi/10 - 0.628) < 0.0005

print("All chapter schrodinger checks passed.")
