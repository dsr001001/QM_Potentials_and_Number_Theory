import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
import qmnt

# Sample: parity / A12
I = quad(lambda u: np.sin(np.pi*u)*np.sin(2*np.pi*u)**2, 0, 1)[0]
A12 = -2*I
assert abs(I-16/(15*np.pi)) < 1e-10 and abs(A12+0.6791) < 1e-4
assert abs(A12**2/3-0.1537) < 1e-4

# Box helpers
def boxA(K, coeffs):
    k = np.arange(1, K+1); kk, ll = np.meshgrid(k, k, indexing="ij")
    A = np.zeros((K, K))
    for j, c in coeffs.items():
        A += c*(-1)**j/2*((abs(kk-ll)==2*j).astype(float)-(kk+ll==2*j).astype(float))
    return k**2.0, A
def d2(E, A):
    D = E[:,None]-E[None,:]; np.fill_diagonal(D, np.inf)
    return -(A**2/D).sum(axis=1)
# verify boxcos formula numerically
L = np.pi/2
psi = lambda k, x: np.sin(k*np.pi*(x+L)/(2*L))/np.sqrt(L)
num = quad(lambda x: psi(5,x)*np.cos(np.pi*x/L)*psi(3,x), -L, L)[0]
assert abs(num-(-0.5)) < 1e-10

# Sample: level 5
E, A = boxA(200, {1:1.0}); s = d2(E, A)
assert abs(s[4]+1/192) < 1e-12
assert abs(s[4]+1/(8*24)) < 1e-12
# Sample: truncated 5 states
E5, A5 = boxA(5, {1:1.0}); s5 = d2(E5, A5)
assert np.allclose(s5, [1/32, 1/48, -1/64, -1/48, -1/64]) and abs(s5.sum()) < 1e-15
assert abs(s[3]+1/120) < 1e-12
# Problem 4 special levels formula check (j=2)
Ej, Aj = boxA(200, {2:1.0}); sj = d2(Ej, Aj)
assert abs(sj[1]-1/(32*4)) < 1e-12 and abs(sj[3]-1/(48*4)) < 1e-12
# truncation figure: factor >15 for top level of 30
E30, A30 = boxA(30, {1:1.0}); s30 = d2(E30, A30)
assert s30[-1]/(-1/(8*(900-1))) > 15

# Sample: residual & Jacobian
T = np.array([[2,1],[1,3.]]); c = np.array([1,-1.])
r = (-0.6+c@T@c)/1.5; J = 2*T@c/1.5
assert abs(r-1.6) < 1e-12 and np.allclose(J, [1.3333, -2.6667], atol=1e-4)
# Problem 5
T = np.array([[1,0.5],[0.5,2]]); c = np.array([2,1.])
assert abs((0.3+c@T@c)/1.2-6.9167) < 1e-4 and np.allclose(2*T@c/1.2, [4.1667,5.0], atol=1e-4)

# Captures
null = 0.2438
for f, p in [(0.2321,4.80),(0.2358,3.28),(0.2322,4.76)]:
    assert abs(100*(null-f)/null-p) < 0.01
assert abs(100*(0.2358-0.2322)/0.2358-1.53) < 0.01
# Problem 1
assert abs(100*(0.2523/null-1)-3.49) < 0.01 and abs(100*(0.2477/null-1)-1.60) < 0.01
# Problem 13
for f, p in [(0.2335,4.22),(0.2422,0.66),(0.2391,1.93),(0.2423,0.62)]:
    assert abs(100*(null-f)/null-p) < 0.01

# float32 overlaps
N = 800; h = 2*L/(N+1)
Ev, ps = eigh_tridiagonal(2/h**2*np.ones(N), -1/h**2*np.ones(N-1), select="i", select_range=(0,59))
ps /= np.sqrt((ps**2).sum(axis=0)*h)
off = ~np.eye(60, dtype=bool)
p32 = ps.astype(np.float32); G32 = (p32.T@p32)*np.float32(h)
m32 = np.median(abs(G32[off])); m64 = np.median(abs((ps.T@ps*h)[off]))
assert 1e-10 < m32 < 1e-6 and m64 < 1e-12
# c0*eta range
assert abs(2.2e6*1e-9-2.2e-3) < 1e-12 and abs(2.2e6*1.2e-7-0.264) < 1e-3

# Harvest sample
y = np.array([0.3,-0.1,0.2,-0.4])
for eps in [1e-7, 1e-3, 1e-9]:
    g = np.ones(4); z = eps*np.array([1,-1,-1,1.])
    b, *_ = np.linalg.lstsq(np.vstack([g, z]).T, y, rcond=None)
    res = y-np.vstack([g, z]).T@b
    assert abs(b[1]*eps+0.05) < 1e-6 and abs(np.sqrt(np.mean(res**2))-0.2693) < 1e-4
assert abs(np.sqrt(0.075)-0.2739) < 1e-4
# harvest figure claim
rng = np.random.default_rng(2); n = 400; yy = 0.24*rng.standard_normal(n)
xx = np.linspace(0,1,n); G = np.array([np.cos(k*np.pi*xx) for k in range(20)]).T
Z = 1e-7*rng.standard_normal((n,200)); X = np.hstack([G,Z[:,:100]])
b, *_ = np.linalg.lstsq(X, yy, rcond=None)
assert 1e4 < np.median(abs(b[20:])) < 1e7

# Block coherence: alternating, ratios (Problems 9, 19)
for a, w in [(0.24,3),(0.24,5)]:
    dd = a*(-1.0)**np.arange(100)
    assert abs(np.sqrt(np.mean(np.convolve(dd,np.ones(w)/w,"valid")**2))-a/w) < 1e-12
for v, rr in zip([0.108,0.049,0.033,0.035,0.019],[2.15,4.73,7.03,6.63,12.2]):
    assert abs(0.232/v-rr) < 0.01
# qmnt block rms (figure, Problem 11)
nn = np.arange(1, 401); dz = qmnt.fluct(qmnt.zeros(400), nn)
for w in [3,5,7,9,13]:
    assert np.sqrt(np.mean(np.convolve(dz,np.ones(w)/w,"valid")**2)) < 0.15
# p=2 period in levels
P = 2*np.pi/np.log(2); assert abs(P-9.065) < 1e-3
g = qmnt.mean_gap(np.array([200, 400]))
assert abs(g[0]-1.516) < 2e-3 and abs(g[1]-1.341) < 2e-3
assert abs(P/g[0]-6.0) < 0.05 and abs(P/g[1]-6.8) < 0.05

# Weak capture
sF = np.sqrt(0.2438**2-0.033**2); assert abs(sF-0.2416) < 1e-4
sr = lambda sm, rho=0.3: np.sqrt(0.2438**2-2*rho*sF*sm+sm**2)
assert abs(sr(0.07)-0.2328) < 1e-4 and abs(sr(0.06)-0.2331) < 1e-4
assert abs(0.3*sF-0.0725) < 1e-4 and abs(sr(0.3*sF)-0.2328) < 1e-4

# Two-level
ex = lambda x: 0.5*(1-np.sqrt(1-4*x**2))
assert abs(ex(0.25)-0.0670) < 1e-4 and abs(ex(0.25)/0.25-0.268) < 1e-3
assert abs(ex(0.3)-0.1) < 1e-12  # Problem 7
w = np.linalg.eigvals(np.array([[0,1j],[1j,1.]])); assert np.allclose(w.real, 0.5)
r4 = (0.16-ex(0.4))/ex(0.4); r1 = (0.01-ex(0.1))/ex(0.1)
assert abs(r4+0.200) < 1e-3 and abs(r1+0.0101) < 1e-4 and abs(r4/r1-19.8) < 0.1
# Problem 15 series check
xs = 0.05; assert abs(ex(xs)-(xs**2+xs**4+2*xs**6)) < 1e-9

# Noise sigma
assert abs(np.sqrt(0.2523**2-null**2)-0.0649) < 1e-4
assert abs(np.sqrt(0.2477**2-null**2)-0.0438) < 1e-4

# Projection and penalty
c = np.array([3,4,12.]); cp = c*min(1,4/np.linalg.norm(c))
assert np.allclose(cp, [0.923,1.231,3.692], atol=1e-3)
Lb = 44.883
assert abs(2*np.pi-3*np.sqrt(3)-1.0870) < 1e-4 and abs(Lb/np.pi-14.287) < 1e-3
pen = quad(lambda x: max(-1+2*np.cos(np.pi*x/Lb),0)**2, -Lb, Lb, points=[-Lb/3,Lb/3])[0]
assert abs(pen-15.53) < 0.01 and abs(1e4*pen-1.553e5) < 200

# Box exact vs surrogate figure claims
K = 80; Eb, _ = boxA(K, {})
cr = np.random.default_rng(1).standard_normal(6)
def rel(s):
    _, A = boxA(K, {j: s*cr[j-1] for j in range(1,7)})
    sec = d2(Eb, A); w, v = np.linalg.eig(np.diag(Eb)+1j*A)
    exs = w[np.argmax(abs(v),axis=1)].real-Eb
    return np.sqrt(np.mean((exs[:40]-sec[:40])**2)/np.mean(sec[:40]**2))
assert rel(0.5) < 0.05 and rel(15) > 0.10
print("all checks passed")
