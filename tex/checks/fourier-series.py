import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import sici
import qmnt

l2, l3 = np.log(2), np.log(3)
# Sample: amplitude and phase
assert abs(np.hypot(3, 4) - 5) < 1e-12 and abs(-np.arctan(4/3) + 0.9273) < 1e-4
# Sample RE2 amplitudes, periods
A = [1/(np.pi*m*2**(m/2)) for m in (1, 2, 3)]
for a, c in zip(A, [0.2251, 0.0796, 0.0375]): assert abs(a - c) < 1e-4
for m, c in zip((1, 2, 3), [9.0647, 4.5324, 3.0216]): assert abs(2*np.pi/(m*l2) - c) < 1e-4
# Sample beats 11,13 and checkpoint
wb, dw = (np.log(11)+np.log(13))/2, np.log(13/11)
assert abs(wb-2.481422) < 1e-6 and abs(dw-0.167054) < 1e-6
assert abs(2*np.pi/wb-2.5321) < 1e-4 and abs(2*np.pi/dw-37.61) < 0.01 and abs(wb/dw-14.85) < 0.01
sil = [(2*j+1)*np.pi/dw for j in range(4)]
assert sum(s < 100 for s in sil) == 3 and abs(sil[3]-131.6) < 0.1
# Orthogonality sample
L = 44.883
I = lambda g: quad(g, -L, L, limit=200)[0]
assert abs(I(lambda x: np.cos(2*np.pi*x/L)*np.cos(3*np.pi*x/L))) < 1e-8
assert abs(I(lambda x: np.cos(2*np.pi*x/L)**2) - L) < 1e-8
# Square wave
assert abs(4/np.pi - 1.2732) < 1e-4 and abs(np.pi**2/8 - 1.2337) < 1e-4
# Gibbs
Si = sici(np.pi)[0]; assert abs(Si-1.851937) < 1e-6 and abs(2/np.pi*Si-1.17898) < 1e-5
K = 50; x = np.pi/(2*K)
S50 = 4/np.pi*sum(np.sin((2*j-1)*x)/(2*j-1) for j in range(1, K+1)); assert abs(S50-1.1790) < 2e-4
assert abs((2/np.pi*Si-1)/2 - 0.0895) < 1e-3
# Complex RE2 and Parseval
for a, c in zip(A, [0.1125, 0.0398, 0.0188]): assert abs(a/2 - c) < 1e-4
ms = sum(a**2 for a in A)/2
assert abs(sum(1/(m**2*2**m) for m in (1, 2, 3)) - 0.576389) < 1e-6
assert abs(ms-0.029200) < 1e-6 and abs(np.sqrt(ms)-0.1709) < 1e-4 and abs(A[0]/np.sqrt(2)-0.1592) < 1e-4
# |x| sample
c0, c1, c3 = L/2, -4*L/np.pi**2, -4*L/(9*np.pi**2)
assert abs(c0-22.4415) < 1e-4 and abs(c1+18.190) < 1e-3 and abs(c3+2.0212) < 1e-4
E = L**2/3 - c0**2 - 0.5*(c1**2+c3**2)
assert abs(E/L**2 - 1.9165e-4) < 1e-7 and abs(np.sqrt(E)-0.621) < 1e-3
assert abs(1-E/(L**2/3) - 0.9994) < 1e-4 and abs(L/np.sqrt(3)-25.91) < 0.01 and abs(np.sqrt(E)/(L/np.sqrt(3))-0.024) < 1e-3
# Basel
assert abs(np.pi**2/6-1.6449) < 1e-4 and abs(np.pi**2/12-0.8225) < 1e-4
# Quasi-periodic
assert abs(l3/l2 - 1.5849625) < 1e-7
t0 = 20.5*np.pi/l2; assert abs(t0-92.913) < 1e-3
assert abs(np.sin(t0*l2)+np.sin(t0*l3) - 1.99967) < 2e-5
for a, tau, r, eps, d in [(1, 9.065, 1.58496, .41504, 1.929), (2, 18.129, 3.16993, .16993, 1.018),
                          (5, 45.324, 7.92481, .07519, .468), (12, 108.777, 19.01955, .01955, .1228),
                          (41, 371.654, 64.98346, .01654, .1039), (53, 480.430, 84.00301, .00301, .0189)]:
    ar = a*l3/l2; e = abs(ar-round(ar))
    assert abs(2*np.pi*a/l2-tau) < 1e-3 and abs(ar-r) < 1e-5 and abs(e-eps) < 1e-5
    assert abs(2*abs(np.sin(np.pi*e)) - d) < 1e-3
assert abs(3**12/2**19 - 1.01364) < 1e-5 and abs(3.0**53/2.0**84 - 1.0021) < 1e-4
# Window sample
cyc = 40*l2/(2*np.pi); assert abs(cyc-4.4127) < 1e-4
sinc = lambda th: np.sin(th)/th
assert abs(abs(sinc(np.pi*(cyc-4)))-0.742) < 1e-3 and abs(abs(sinc(np.pi*(cyc-5)))-0.522) < 1e-3
assert abs(8*np.pi/l2-36.26) < 0.01
# Paper connection
assert abs(2*L/399-0.225) < 1e-3 and abs(399*np.pi/L-27.93) < 0.01
assert abs(2*np.sqrt(811.18)*L/np.pi-813.8) < 0.1 and abs(400*np.pi/L-28.0) < 0.01 and abs((400*np.pi/L/2)**2-196) < 1
Nsm = lambda t: t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
sz = lambda n: brentq(lambda t: Nsm(t)-(n-0.5), 2*np.pi+1e-9, 1e5)
gp = lambda s: 2*np.pi/np.log(s/(2*np.pi))
s400, s500 = sz(400), sz(500)
assert abs(s400-679.9) < 0.1 and abs(gp(s400)-1.341) < 1e-3 and abs(2*np.pi/l2/gp(s400)-6.76) < 0.01
assert abs(gp(s500)-1.293) < 1e-3 and abs(2*np.pi/l2/gp(s500)-7.01) < 0.01
assert abs(np.asarray(qmnt.smooth_zeros(np.array([400])))[0]-s400) < 1e-6
sn = np.array([sz(n) for n in range(401, 501)])
m = np.arange(1, 4)
re2 = -np.sum(np.sin(np.outer(sn, m*l2))/(m*2**(m/2)), axis=1)/np.pi
r = np.sqrt(np.mean(re2**2)); assert abs(r-0.1709) < 0.003 and abs(r-0.1708) < 0.003
assert abs((sn[-1]-sn[0])/(2*np.pi/l2) - 14.3) < 0.2
z = qmnt.zeros(1000)
assert abs(2*np.pi*450/8-353.43) < 0.01
assert 326 < z[400]-353.43 < 330 and abs(z[499]-353.43-457.8) < 0.1 and abs(z[399]-680) < 2
# Problems
assert abs(2*np.pi/l3-5.719) < 1e-3                                   # P1
assert abs(np.arctan(12/5)-1.176) < 1e-3                              # P2
assert abs(2*np.pi/np.log(1.4)-18.674) < 1e-3                         # P3
assert abs(np.pi**4/96-1.01468) < 1e-5                                # P9
assert abs(0.5/0.576389-0.8675) < 1e-4                                # P11
assert abs(2*59.957/899-0.1334) < 1e-4                                # P14
assert abs(2*np.sqrt(1419.42)*59.957/np.pi-1438.1) < 0.2               # P16
assert abs(2*np.pi/l3/gp(s400)-4.264) < 1e-3                          # P18
tot = sum(1/(mm**2*p**mm) for p in (2, 3, 5, 7, 11, 13) for mm in (1, 2, 3))
assert abs(tot-1.472299) < 1e-6 and abs(tot/(2*np.pi**2)-0.074588) < 1e-6 and abs(np.sqrt(tot/(2*np.pi**2))-0.2731) < 1e-4  # P19
assert abs(1-8/np.pi**2-0.1894) < 1e-4                                # P21
assert abs(3*np.log(13)-7.695) < 1e-3 and 3**3*0+13**3 == 2197        # P26
assert abs(6/(2*L/399)-26.7) < 0.05                                   # P27
assert abs(2/np.pi-0.6366) < 1e-4 and abs(1-8/np.pi**2-0.189) < 1e-3  # P29
# P17: numerical complex coefficients of RE2 over one period
Tp = 2*np.pi/l2; tt = np.linspace(0, Tp, 20001)
f = -sum(np.sin(mm*tt*l2)/(mm*2**(mm/2)) for mm in (1, 2, 3))/np.pi
for k, c in [(1, 0.1125), (2, 0.0398), (3, 0.0188)]:
    fk = np.trapezoid(f*np.exp(-1j*k*l2*tt), tt)/Tp
    assert abs(fk.real) < 1e-6 and abs(fk.imag - c) < 1e-4
print("all checks passed")
