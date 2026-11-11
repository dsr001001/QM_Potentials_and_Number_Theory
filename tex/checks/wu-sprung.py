import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

V0 = 3.10073*np.pi
def Nsm(t):
    u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def xws(V, V0=V0):
    a = np.sqrt(V-V0); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b+a)/(b-a)))/np.pi
def sn(n):
    return brentq(lambda t: Nsm(t)-(n-0.5), 7.0, 5000.0)

# Sample: lopsided oscillator
beta = 0.4
for E in [1.0, 3.0, 7.0]:
    ph = quad(lambda x: np.sqrt(max(E-x**2/(1+beta)**2, 0)), 0, (1+beta)*np.sqrt(E))[0] \
       + quad(lambda x: np.sqrt(max(E-x**2/(1-beta)**2, 0)), -(1-beta)*np.sqrt(E), 0)[0]
    assert abs(ph - np.pi*E/2) < 1e-6

# Sample: density at 1000
u = 1000/(2*np.pi)
assert abs(u-159.155) < 1e-3 and abs(np.log(u)-5.0699) < 1e-4
assert abs(np.log(u)/(2*np.pi)-0.8069) < 1e-4
assert abs(2*np.pi/np.log(u)-1.239) < 1e-3
assert abs(Nsm(1000)-648.6) < 0.05

# Sample: power-law integrals
assert abs(quad(lambda v: np.sqrt(1-v**2), 0, 1)[0]-np.pi/4) < 1e-8
assert abs(quad(lambda v: np.sqrt(1-v), 0, 1)[0]-2/3) < 1e-8

# Kernel K = pi (V=2, W=5), pycode listing
K = quad(lambda E: 1/np.sqrt((5-E)*(E-2)), 2, 5)[0]
assert abs(K-np.pi) < 1e-9

# Sample: harmonic inversion
V = 7.0
xv = quad(lambda E: 0.5/np.sqrt(V-E), 0, V)[0]
assert abs(xv-np.sqrt(V)) < 1e-6

# Closed form Eq.(2) vs integral
for V in [50.0, 500.0]:
    num = quad(lambda uu: np.log((V-uu**2)/(2*np.pi)), 0, np.sqrt(V-V0))[0]/np.pi
    assert abs(num - xws(V)) < 1e-8

# Abel uniqueness: numerical instance, Phi_WS = pi(Nsm(E)-Nsm(V0)) at E=300
E = 300.0
PhiWS = 2*quad(lambda uu: xws(E-uu**2) if E-uu**2 > V0 else 0.0, 0, np.sqrt(E-V0), limit=200)[0]
assert abs(PhiWS - np.pi*(Nsm(E)-Nsm(V0))) < 1e-5

# Sample: V0 root; limits of f
f = lambda y: y*np.log(y)-y+7/8
assert f(1.5) < 0 and f(1.6) > 0
assert abs(1.5*(np.log(1.5)-1)+0.8918) < 1e-4 and abs(1.6*(np.log(1.6)-1)+0.8480) < 1e-4
y = brentq(f, 1.0, 3.0)
assert abs(y-1.54013) < 1e-5
assert abs(2*y-3.0803) < 1e-3
assert abs(Nsm(V0)-0.0045) < 1e-4
yl = brentq(f, 0.05, 1.0); assert yl < 1
yy = 1e-8; assert abs(yy*np.log(yy) - yy) < 1e-6   # f-7/8 -> 0 as y->0+

# Sample: harmonic bottom
lg = np.log(V0/(2*np.pi))
assert abs(lg-0.43849) < 1e-5 and abs(np.pi/lg-7.1645) < 1e-3
assert abs((np.pi/lg)**2-51.33) < 0.02
Vt = V0+1e-4
assert abs(xws(Vt) - np.sqrt(1e-4)/np.pi*lg) < 1e-6

# Sample: box sizes, with intermediates
assert abs(V0-9.7412) < 1e-3
assert abs((lg-2)+1.56151) < 1e-5
a = np.sqrt(1000-V0); b = np.sqrt(1000)
assert abs(1000-V0-990.259) < 1e-3
assert abs(a-31.468) < 1e-3 and abs(b-31.623) < 1e-3
assert abs(b+a-63.091) < 1e-3
assert abs(V0/(b+a)-0.1544) < 1e-4 and abs((b-a)-0.1544) < 1e-4
lr = np.log((b+a)/(b-a)); assert abs(lr-6.0128) < 1e-4
assert abs(a*(lg-2)+49.14) < 0.01 and abs(b*lr-190.14) < 0.01
assert abs(a*(lg-2)+b*lr-141.00) < 0.01
assert abs(xws(1000.0)-44.883) < 1e-3
assert abs(xws(1500.0)-59.957) < 1e-3
asym = np.sqrt(1000)/np.pi*(np.log(2000/np.pi)-2)
assert abs(asym-44.855) < 1e-3
rel = (xws(1000.0)-asym)/xws(1000.0)
assert abs(rel-0.0006) < 5e-5
try:
    import qmnt
    assert abs(qmnt.L_of(1000)-44.883) < 1e-3
    assert abs(qmnt.L_of(1500)-59.957) < 1e-3
except ImportError:
    pass
# s_500 approx 811
assert abs(sn(500)-811.5) < 0.6

# Sample: turning points 201, 300 (and Problem 9, Paper connection)
s201, s300 = sn(201), sn(300)
print('s201, s300 =', s201, s300)
assert abs(s201-398.3) < 0.05
assert abs(s300-542.5) < 0.1
assert abs(Nsm(400.0)-201.6) < 0.1
assert abs(np.log(400/(2*np.pi))/(2*np.pi)-0.661) < 1e-3
assert abs((Nsm(400.0)-200.5)/0.661-1.7) < 0.05
assert abs(xws(s201)-22.5) < 0.05
assert abs(xws(s300)-28.5) < 0.06
assert abs(s201-398) < 0.5 and abs(s300-543) < 0.6
# Checkpoint: x_500 about 38.5
assert abs(xws(sn(500))-38.5) < 0.1

# Check 1 / Problem 13 (n = 1, 2, 10, 100, 250, 500 as stated in text)
for n in [1, 2, 10, 100, 250, 500]:
    E = sn(n)
    Phi = 2*quad(lambda uu: xws(E-uu**2) if E-uu**2 > V0 else 0.0, 0, np.sqrt(E-V0), limit=200)[0]
    assert abs(Phi/np.pi-(n-0.5)+Nsm(V0)) < 1e-6

# Check 2: Richardson FD levels vs smooth zeros (same as listing: table + interp)
aa = np.linspace(0, np.sqrt(1200-V0), 400001); VV = V0+aa**2
XX = np.concatenate([[0.0], xws(VV[1:])])
L = xws(1000.0)
def levels(N, k=500):
    x = -L + 2*L/(N+1)*np.arange(1, N+1); h = 2*L/(N+1)
    Vv = np.interp(np.abs(x), XX, VV)
    return eigh_tridiagonal(2/h**2+Vv, -np.ones(N-1)/h**2, select='i',
                            select_range=(0, k-1), eigvals_only=True)
ER = (4*levels(23999)-levels(11999))/3
ns = np.arange(1, 501)
s = np.array([sn(k) for k in ns])
g = 2*np.pi/np.log(s/(2*np.pi))
dev = (ER-s)/g
print('mean dev, max |dev| =', np.mean(dev), np.max(np.abs(dev)), np.argmax(np.abs(dev)))
assert abs(np.mean(dev)-0.004) < 0.0006          # signed mean 0.004
assert abs(np.max(np.abs(dev))-0.05) < 0.005      # max 0.05
assert np.argmax(np.abs(dev)) < 50                # attained at lowest levels

# Problem 2
assert abs(100/(2*np.pi)-15.9155) < 1e-4
assert abs(np.log(100/(2*np.pi))-2.7673) < 1e-4
assert abs(2*np.pi/np.log(100/(2*np.pi))-2.2705) < 1e-4
# Problem 7
assert abs((1*np.log(1)-1)+1) < 1e-12
# Problem 14
per = 2*np.pi/np.log(2); gp = 2*np.pi/np.log(500/(2*np.pi))
assert abs(per-9.065) < 1e-3 and 6 <= per/gp <= 7
assert abs(sn(400)-680) < 1.0
g680 = 2*np.pi/np.log(680/(2*np.pi)); assert 6 <= per/g680 <= 7
# Problem 17
Vq = 3.0
assert abs((2/np.pi)*quad(lambda E: np.sqrt(E)/np.sqrt(Vq-E), 0, Vq)[0]-Vq) < 1e-6
# Problem 19
t1 = 14.134725
assert abs(t1/(2*np.pi)-2.24961) < 1e-5 and abs(np.log(t1/(2*np.pi))-0.81076) < 1e-5
assert abs(Nsm(t1)-0.449) < 1e-3
assert abs((0.5-Nsm(t1))-0.05) < 0.005
s1 = sn(1); assert t1 < s1
print('all checks passed')
