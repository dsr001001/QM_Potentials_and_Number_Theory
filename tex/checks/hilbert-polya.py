import numpy as np, qmnt

# RE1: first ten zeros
RE1 = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
       37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
z10 = qmnt.zeros(10)
assert np.allclose(z10, RE1, atol=2e-6)

# Sample "Relabeling a zero"
tf = lambda rho: -1j*(rho - 0.5)
assert abs(tf(0.5 + 14.134725j) - 14.134725) < 1e-12
assert abs(tf(0.6 + 20j) - (20 - 0.1j)) < 1e-12
# Problem 1
assert abs(tf(0.5 + 21.022040j) - 21.022040) < 1e-12
assert abs(tf(0.55 + 30j) - (30 - 0.05j)) < 1e-12

# Sample "Real eigenvalues from Hermiticity"
H = np.array([[2, 1-1j], [1+1j, 3]])
assert np.allclose(np.sort(np.linalg.eigvalsh(H)), [1, 4])
B = np.array([[1., 2.], [-1., 1.]])
ev = np.linalg.eigvals(B)
assert np.allclose(sorted(ev, key=lambda z: z.imag), [1 - 1j*np.sqrt(2), 1 + 1j*np.sqrt(2)])
assert np.allclose(np.sort(np.linalg.eigvals(np.array([[1., 1.], [0., 2.]])).real), [1, 2])
# Problem 2
H2 = np.array([[3, 2j], [-2j, 3]])
assert np.allclose(np.sort(np.linalg.eigvalsh(H2)), [1, 5])

# Sample "A Hermitian-looking operator with complex eigenvalues"
from scipy.integrate import quad
k = 2 + 1j
assert abs(quad(lambda x: abs(np.exp(1j*k*x))**2, 0, np.inf)[0] - 0.5) < 1e-10
x0, h = 0.7, 1e-6
f = lambda x: np.exp(1j*k*x)
assert abs(-1j*(f(x0+h) - f(x0-h))/(2*h) - k*f(x0)) < 1e-6

# Sample "An operator for two zeros" and Problem 21
t1, t2, t3 = RE1[0], RE1[1], RE1[2]
c, s = np.sqrt(3)/2, 0.5
assert abs(c*s - 0.4330127) < 1e-7
assert abs((t1 - t2) - (-6.887315)) < 1e-9
R = np.array([[c, -s], [s, c]])
H2z = R @ np.diag([t1, t2]) @ R.T
assert abs(H2z[0, 0] - 15.856554) < 2e-6
assert abs(H2z[1, 1] - 19.300211) < 2e-6
assert abs(H2z[0, 1] - (-2.982295)) < 2e-6
assert abs(np.trace(H2z) - 35.156765) < 1e-6
assert np.allclose(np.linalg.eigvalsh(H2z), [t1, t2])
H3 = np.zeros((3, 3)); H3[:2, :2] = H2z; H3[2, 2] = t3
assert np.allclose(np.linalg.eigvalsh(H3), [t1, t2, t3])

# Sample "Counting states in the paper's box" (RE3)
L = 44.883
assert abs(qmnt.L_of(1000) - L) < 1e-3
Nrule = 2*L*np.sqrt(1000)/np.pi
assert abs(Nrule - 903.57) < 0.01
assert abs(np.sqrt(1000) - 31.6228) < 1e-4
kk = np.arange(1, 2000)
assert np.sum((kk*np.pi/(2*L))**2 <= 1000) == 903
# Problem 4
assert abs(10*10/np.pi - 31.83) < 0.01
assert np.sum((kk*np.pi/10)**2 <= 100) == 31

# Sample "Summing the orbit series at E=9/4"
def osc(E, R):
    r = np.arange(1, R+1)
    return E + np.sum((-1.0)**r*np.sin(2*np.pi*r*E)/r)/np.pi
assert abs(osc(2.25, 1) - 1.931690) < 1e-6
assert abs(osc(2.25, 3) - 2.037793) < 1e-6
assert abs(osc(2.25, 5) - 1.974131) < 1e-6
assert abs(1/(3*np.pi) - 0.106103) < 1e-6 and abs(1/(5*np.pi) - 0.063662) < 1e-6
assert abs(osc(2.25, 200001) - 2.0) < 1e-5
assert abs(np.floor(2.25 + 0.5) - 2) < 1e-12
# Leibniz series: two million terms vs pi/4 = 0.785398
j = np.arange(0, 2000000)
leib = np.sum((-1.0)**j/(2*j+1))
assert abs(leib - np.pi/4) < 1e-6
assert abs(np.pi/4 - 0.785398) < 1e-6
# Problem 6: E = 7/4
assert abs(osc(1.75, 200001) - 2.0) < 1e-5
# Problem 18: error bound for odd R
for Rr in range(1, 200, 2):
    assert abs(osc(2.25, Rr) - 2.0) <= 1/(np.pi*Rr) + 1e-12

# Sample "The p=2 orbit"
assert abs(np.log(2) - 0.693147) < 1e-6
assert abs(2*np.pi/np.log(2) - 9.064720) < 1e-6
assert abs(1/(np.pi*np.sqrt(2)) - 0.225079) < 1e-6
assert abs(np.sqrt(2) - 1/np.sqrt(2) - 0.707107) < 1e-6
assert abs(np.sqrt(2)/(np.sqrt(2) - 1/np.sqrt(2)) - 2) < 1e-12
assert abs(2*np.log(3) - 2.197) < 1e-3
# Problem 9
assert abs(1/(1 - 2.0**-3) - 8/7) < 1e-12 and abs(8/7 - 1.143) < 1e-3
assert abs(1/(1 - 1/13) - 13/12) < 1e-12 and abs(13/12 - 1.083) < 1e-3
# det expansion check
for Lam in [1.7, 2.0, 5.0]:
    for r in range(1, 4):
        assert abs((Lam**r - 1)*(Lam**-r - 1) - (2 - Lam**r - Lam**-r)) < 1e-9
        assert abs((2 - Lam**r - Lam**-r) + (Lam**(r/2) - Lam**(-r/2))**2) < 1e-9

# Problem 5
amp = {p: np.log(p)/np.sqrt(p) for p in [2, 3, 5, 7, 11, 13]}
for p, v in zip([2, 3, 5, 7, 11, 13], [0.490, 0.634, 0.720, 0.735, 0.723, 0.711]):
    assert abs(amp[p] - v) < 1e-3
assert max(amp, key=amp.get) == 7
yy = np.linspace(1.5, 30, 200001)
assert abs(yy[np.argmax(np.log(yy)/np.sqrt(yy))] - np.e**2) < 1e-3

# Problem 10
for mu in range(-20, 21):
    ok = all(abs(np.exp(-1j*r*mu*np.pi/2) + 1) < 1e-9 for r in (1, 2))
    assert not ok

# Sample "Berry-Keating count at E=1000"
def Nsm(E): return E/(2*np.pi)*np.log(E/(2*np.pi)) - E/(2*np.pi) + 7/8
def NBK(E): return E/(2*np.pi)*np.log(E/(2*np.pi)) - E/(2*np.pi) + 1
assert abs(1000/(2*np.pi) - 159.15494) < 1e-5
assert abs(np.log(1000/(2*np.pi)) - 5.069878) < 1e-6
assert abs(1000/(2*np.pi)*(np.log(1000/(2*np.pi)) - 1) - 647.7412) < 1e-3
assert abs(Nsm(1000) - 648.6162) < 1e-3 and abs(Nsm(1000) - 648.62) < 0.005
assert abs(NBK(1000) - 648.7412) < 1e-3 and abs(NBK(1000) - 648.74) < 0.005
assert round(Nsm(1000.0), 3) == 648.616 and round(NBK(1000.0), 3) == 648.741
assert abs(NBK(1000) - Nsm(1000) - 0.125) < 1e-12
assert abs(qmnt.Nsm(1000.0) - Nsm(1000.0)) < 1e-9
# Box sample: 652 states (paper) vs Nsm(1000): about three levels
assert abs((652 - Nsm(1000)) - 3.4) < 0.1
z700 = qmnt.zeros(700)
assert np.sum(z700 <= 1000) == 649
assert abs((649 - Nsm(1000)) - 0.38) < 0.005
lx = lp = np.sqrt(2*np.pi); E = 40.0
area = quad(lambda x: E/x - lp, lx, E/lp)[0]
assert abs(area/(2*np.pi) - NBK(E)) < 1e-9
# Problem 7: general a
a = 3.0
area_a = quad(lambda x: E/x - 1.0, a, E/1.0)[0]
claim = E/(2*np.pi)*np.log(E/(2*np.pi)) - E/(2*np.pi) + E/(2*np.pi)*np.log(2*np.pi/a) + a/(2*np.pi)
assert abs(area_a/(2*np.pi) - claim) < 1e-9

# Sample "Orbit time and spacing at E=1000"
assert abs(np.log(1000/(2*np.pi)) - 5.0699) < 1e-4
assert abs(2*np.pi/np.log(1000/(2*np.pi)) - 1.239317) < 1e-6
dEh = 1e-3
assert abs((NBK(1000+dEh) - NBK(1000-dEh))/(2*dEh) - np.log(1000/(2*np.pi))/(2*np.pi)) < 1e-8
# Problem 8
t500 = qmnt.zeros(500)[-1]
assert abs(t500 - 811.1843588) < 1e-6
assert abs(2*np.pi/np.log(t500/(2*np.pi)) - 1.293) < 1e-3

# Sample "Eigenfunctions of the quantum xp" and Problem 11
Ev = 3.7
psi = lambda x: x**(-0.5 + 1j*Ev)
xx, hh = 2.3, 1e-6
dpsi = (psi(xx+hh) - psi(xx-hh))/(2*hh)
assert abs(-1j*(xx*dpsi + psi(xx)/2) - Ev*psi(xx)) < 1e-6
assert abs(abs(psi(xx))**2 - 1/xx) < 1e-12
assert abs(quad(lambda x: abs(psi(x))**2, 1, 100.0)[0] - np.log(100.0)) < 1e-7

# Problem 19
w = 1.7; Eo = 5.0
Aell = 2*quad(lambda x: np.sqrt(max(Eo - w**2*x**2/4, 0.0)), -2*np.sqrt(Eo)/w, 2*np.sqrt(Eo)/w)[0]
assert abs(Aell - 2*np.pi*Eo/w) < 1e-6
dA = (2*np.pi*(Eo+1e-4)/w - 2*np.pi*(Eo-1e-4)/w)/2e-4
assert abs(dA - 2*np.pi/w) < 1e-8

# Problem 20
al = 0.9
M = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
for r in range(1, 5):
    Mr = np.linalg.matrix_power(M, r)
    assert abs(np.linalg.det(Mr - np.eye(2)) - 4*np.sin(r*al/2)**2) < 1e-12
Lam = 2.0
for r in range(1, 5):
    Mr = np.diag([Lam**r, Lam**-r])
    assert abs(abs(np.linalg.det(Mr - np.eye(2)))**0.5 - (Lam**(r/2) - Lam**(-r/2))) < 1e-12

# Problem 15
assert abs(0.074/0.263 - 0.281) < 1e-3
assert abs(0.119/0.296 - 0.402) < 1e-3
assert abs(0.043/0.263 - 0.163) < 1e-3
assert abs(0.067/0.296 - 0.226) < 1e-3
# Problem 14
assert abs(np.log10(1e-3/2.6e-55) - 51.6) < 0.05

# Problem 17
T = np.linspace(100.0, 1000.0, 90001)
NT = np.searchsorted(z700, T, side="right")
m1 = np.mean(NT - Nsm(T)); m2 = np.mean(NT - NBK(T))
assert abs(m1) < 0.05
assert abs(m2 + 0.125) < 0.05

# Paper connection: Table V six-prime row on 401-500 (RE5)
n = np.arange(401, 501)
true = qmnt.zeros(500)[n - 1]
null = qmnt.rms(qmnt.fluct(true, n))
assert abs(null - 0.2631) < 5e-4
pred = qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3)
rms6, corr6 = qmnt.score(pred, n)
assert abs(rms6 - 0.074) < 0.0015 and abs(corr6 - 0.96) < 0.01

print("all checks passed")
