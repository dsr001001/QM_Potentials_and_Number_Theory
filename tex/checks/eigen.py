import numpy as np
from scipy.linalg import eigh_tridiagonal

# Section 1: A = [[2,1],[1,2]]
A = np.array([[2., 1.], [1., 2.]])
assert np.allclose(np.linalg.eigvalsh(A), [1, 3])

# Sample: non-symmetric B
B = np.array([[4., 1.], [2., 3.]])
assert np.allclose(sorted(np.linalg.eigvals(B).real), [2, 5])
assert np.allclose(B @ [1, 1], [5, 5]) and np.allclose(B @ [1, -2], [2, -4])

# Sample: D3
D3 = 2*np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1)
w = np.linalg.eigvalsh(D3)
assert np.allclose(w, [2-np.sqrt(2), 2, 2+np.sqrt(2)])
assert abs(w[0]-0.585786) < 1e-6 and abs(w[2]-3.414214) < 1e-6
for lam, v in [(2-np.sqrt(2), [1, np.sqrt(2), 1]), (2, [1, 0, -1]), (2+np.sqrt(2), [1, -np.sqrt(2), 1])]:
    assert np.allclose(D3 @ np.array(v), lam*np.array(v))
# n x n: det = product, trace = sum
assert abs(np.prod(w) - np.linalg.det(D3)) < 1e-12 and abs(np.sum(w) - np.trace(D3)) < 1e-12

# R90
assert np.allclose(sorted(np.linalg.eigvals(np.array([[0, -1], [1, 0]])).imag), [-1, 1])

# Sample: C rotation+stretch
C = np.array([[1., -2.], [2., 1.]])
lc = np.linalg.eigvals(C)
assert np.allclose(sorted(lc.imag), [-2, 2]) and np.allclose(lc.real, 1)
assert abs(abs(lc[0]) - 2.236) < 1e-3
assert abs(np.degrees(np.arctan(2)) - 63.43) < 0.01
assert np.allclose(C @ np.array([1, -1j]), (1+2j)*np.array([1, -1j]))

# Sample: complex Hermitian
H = np.array([[2, 1j], [-1j, 2]])
assert np.allclose(np.linalg.eigvalsh(H), [1, 3])
u = np.array([1j, 1]); ww = np.array([-1j, 1])
assert np.allclose(H @ u, 3*u) and np.allclose(H @ ww, ww)
assert abs(np.vdot(u, ww)) < 1e-14 and abs(np.dot(u, ww) - 2) < 1e-14

# Sample: A^10
A10 = np.linalg.matrix_power(A.astype(np.int64), 10)
assert (A10 == np.array([[29525, 29524], [29524, 29525]])).all()

# Sample: box RE3
L, N = 44.883, 6000
q = np.pi/(2*L)
assert abs(q - 0.0349976) < 1e-7
assert abs(q**2 - 1.2248e-3) < 1e-7
assert abs(4*q**2 - 4.8993e-3) < 1e-7
assert abs(9*q**2 - 1.10235e-2) < 1e-6
h = 2*L/(N+1)
assert abs(h - 0.014958) < 1e-6
ex500 = (500*q)**2
assert abs(500*q - 17.4988) < 1e-4 and abs(ex500 - 306.208) < 1e-3
ang = 500*np.pi*h/(4*L)
assert abs(ang - 0.130878) < 1e-6
disc = lambda k: 4/h**2*np.sin(k*np.pi*h/(4*L))**2
assert abs(disc(500) - 304.463) < 1e-3
d = 2/h**2*np.ones(N); e = -np.ones(N-1)/h**2
E = eigh_tridiagonal(d, e, select='i', select_range=(0, 499), eigvals_only=True)
assert abs(E[499] - 304.463) < 1e-3
diff = ex500 - disc(500); sp = q**2*1001
assert abs(diff - 1.744) < 1e-3 and abs(sp - 1.2261) < 1e-4 and abs(diff/sp - 1.42) < 0.005

# Table eigen-box
for k, rel, mat in [(1, 2.28e-8, None), (10, 2.28e-6, None), (100, 2.28e-4, 12.2455), (500, 5.70e-3, None)]:
    r = 1 - disc(k)/(k*q)**2
    assert abs(r - rel)/rel < 0.01, (k, r)
    if mat: assert abs(disc(k) - mat) < 1e-4
assert abs(10**2*q**2 - 0.12248) < 1e-5 and abs(100**2*q**2 - 12.2483) < 1e-4

# Figure spectrum: ratio at k=N=600 about 0.41
Nf = 600; hf = 2*L/(Nf+1)
ratio = (4/hf**2*np.sin(Nf*np.pi*hf/(4*L))**2)/(Nf*q)**2
assert abs(ratio - 0.41) < 0.01

# Largest eigenvalue ~ 4/h^2 ~ 1.79e4; rounding ~ few 1e-12
assert abs(4/h**2 - 1.79e4) < 50
assert 1e-12 < 4/h**2*2.2e-16 < 1e-11

# Sturm count sample
def sturm_count(d, b, x):
    c = 0; qq = d[0]-x
    for j in range(len(d)):
        if j > 0: qq = d[j]-x-b[j-1]**2/qq
        if qq == 0.0: qq = -1e-300
        if qq < 0: c += 1
    return c
assert sturm_count(np.full(3, 2.), np.full(2, -1.), 2.5) == 2
assert sturm_count(np.full(3, 2.), np.full(2, -1.), 0.6) == 1
assert abs((1.4 - 1/1.4) - 0.685714) < 1e-6
assert abs((1.4 - 1/(1.4-1/1.4)) - (-0.058333)) < 1e-6
assert abs(-0.5 - 1/1.5 - (-1.16667)) < 1e-5

# Matching sample
Ha = np.array([[1, .5], [.5, 3]])
wa, Ua = np.linalg.eigh(Ha)
assert np.allclose(wa, [0.881966, 3.118034], atol=1e-6)
assert abs(abs(Ua[0, 0])**2 - 0.947214) < 1e-6
assert abs(abs(Ua[1, 0])**2 - 0.052786) < 1e-6
Hb = np.array([[1, .5j], [.5j, 3]])
lb, Wb = np.linalg.eig(Hb)
o = np.argsort(lb.real); lb = lb[o]; Wb = Wb[:, o]
assert np.allclose(lb, [1.133975, 2.866025], atol=1e-6)
O = np.abs(Wb)**2/np.sum(np.abs(Wb)**2, axis=0)
assert abs(O[0, 0]-0.933013) < 1e-6 and abs(O[1, 1]-0.933013) < 1e-6  # O_11 for u, O_22 for w
uu = np.array([1, -0.267949j]); wv = np.array([1, -3.732051j])
assert abs(np.vdot(uu, wv) - 2) < 1e-5
assert abs(abs(np.vdot(uu, wv))/np.linalg.norm(uu)/np.linalg.norm(wv) - 0.5) < 1e-5
assert abs(1 - 0.25/2 - 0.875) < 1e-12 and abs(1 + 0.25/2 - 1.125) < 1e-12

# Problem 15 and 30: g=1.5
lg = np.linalg.eigvals(np.array([[1, 1.5j], [1.5j, 3]]))
assert np.allclose(lg.real, 2) and np.allclose(sorted(lg.imag), [-1.118034, 1.118034], atol=1e-6)

# Tridiagonal storage
assert 2*6000-1 == 11999 and 6000**2 == 3.6e7

# Problem 1, 3, 4
assert np.allclose(np.linalg.eigvalsh(np.array([[3, 2], [2, 3]])), [1, 5])
assert np.allclose(np.linalg.eigvalsh(np.array([[0, 1], [1, 0]])), [-1, 1])
assert np.allclose(np.roots([1, -6, 8]), [4, 2])
# Problem 2
assert np.allclose(sorted(np.linalg.eigvals(np.array([[1, 4, 5], [0, 3, 6], [0, 0, -2]])).real), [-2, 1, 3])
# Problem 7
P7 = np.array([[1, 1+1j], [1-1j, 2]])
assert np.allclose(np.linalg.eigvalsh(P7), [0, 3])
assert abs(np.vdot(np.array([-(1+1j), 1]), np.array([1+1j, 2]))) < 1e-14
# Problem 9: numerical instance theta = 0.7
th = 0.7
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
assert np.allclose(np.sort_complex(np.linalg.eigvals(R)), np.sort_complex(np.array([np.exp(-1j*th), np.exp(1j*th)])))
assert np.allclose(R @ np.array([1, -1j]), np.exp(1j*th)*np.array([1, -1j]))
assert np.allclose(R @ np.array([1, 1j]), np.exp(-1j*th)*np.array([1, 1j]))
# Problem 11
U = np.column_stack([np.array([1, np.sqrt(2), 1])/2, np.array([1, 0, -1])/np.sqrt(2), np.array([1, -np.sqrt(2), 1])/2])
assert np.allclose(U.T @ U, np.eye(3))
assert np.allclose(U.T @ D3 @ U, np.diag([2-np.sqrt(2), 2, 2+np.sqrt(2)]))
# Problem 13
x = np.array([1., 2.]); assert abs(x @ A @ x / (x @ x) - 2.8) < 1e-12
# Problem 15 g=0.5
assert np.allclose(2 + np.array([-1, 1])*np.sqrt(0.75), [1.133975, 2.866025], atol=1e-6)
# Problem 17
w50 = eigh_tridiagonal(2*np.ones(50), -np.ones(49), eigvals_only=True)
k50 = np.arange(1, 51)
assert np.max(np.abs(w50 - (2-2*np.cos(k50*np.pi/51)))) < 1e-13
# Problem 19
N19 = 2000; h19 = 20/(N19+1); x19 = -10 + h19*np.arange(1, N19+1)
E19 = eigh_tridiagonal(2/h19**2 + x19**2, -np.ones(N19-1)/h19**2, select='i', select_range=(0, 4), eigvals_only=True)
err19 = np.abs(E19 - np.array([1, 3, 5, 7, 9]))
assert np.max(err19) < 5e-4 and err19[4] > err19[0]
# Problem 21
assert abs(2*L*np.sqrt(1000)/np.pi - 903.6) < 0.05
assert abs(h/2*np.sqrt(1000) - 0.23651) < 1e-5
assert abs(12002*np.arcsin(h/2*np.sqrt(1000))/np.pi - 912.2) < 0.05
kk = np.arange(1, N+1)
assert np.sum((kk*q)**2 < 1000) == 903 and np.sum(disc(kk) < 1000) == 912
assert sturm_count(d, -np.ones(N-1)/h**2, 1000.0) == 912
# Problem 23
assert 6000**3 == 2.16e11 and 2500**3 == 1.5625e10
# Problem 24
O24 = np.array([[.7, .55, .05], [.25, .4, .15], [.05, .05, .8]])
assert np.allclose(O24.sum(axis=0), 1)
assert list(np.argmax(O24, axis=0)) == [0, 0, 2]
# Problem 25
D4 = 2*np.eye(4) - np.eye(4, k=1) - np.eye(4, k=-1)
x4 = np.array([1., 2., -1., 3.])
assert np.allclose(D4 @ x4, [0, 4, -7, 7]) and abs(x4 @ D4 @ x4 - 36) < 1e-12
assert abs(x4[0]**2 + x4[-1]**2 + np.sum(np.diff(x4)**2) - 36) < 1e-12
# Problem 27 (answer quotes 0.0713 at eps=0.5; exact 0.07125)
for ep, ov in [(-2, 0.9945), (0, 0.5), (0.5, 0.0713), (2, 0.0055)]:
    wq, Uq = np.linalg.eigh(np.array([[ep, .3], [.3, -ep]]))
    assert abs(abs(Uq[0, 0])**2 - ov) < 1e-4, (ep, abs(Uq[0, 0])**2)
    assert abs(0.5*(1 - ep/np.sqrt(ep**2+0.09)) - ov) < 1e-4
    assert abs((wq[1]-wq[0]) - 2*np.sqrt(ep**2+0.09)) < 1e-12
assert abs(2*0.3 - 0.6) < 1e-12
# Problem 29: numerical instance
rng = np.random.default_rng(0)
M = rng.normal(size=(4, 4)); S = rng.normal(size=(4, 4))
assert np.allclose(np.sort_complex(np.linalg.eigvals(M)), np.sort_complex(np.linalg.eigvals(np.linalg.inv(S) @ M @ S)))
# Problem 31
g = 0.4
H31 = np.diag([1., 2., 3.]) + 1j*g*(np.eye(3, k=1) + np.eye(3, k=-1))
l31, W31 = np.linalg.eig(H31)
o = np.argsort(l31.real); l31 = l31[o]; W31 = W31[:, o]
assert np.allclose(l31, [1.175379, 2, 2.824621], atol=1e-6)
assert abs(np.linalg.det(H31) - (6+4*g**2)) < 1e-12
O31 = np.abs(W31)**2/np.sum(np.abs(W31)**2, axis=0)
assert abs(O31[0, 0]-0.832) < 1e-3 and abs(O31[1, 1]-0.758) < 1e-3 and abs(O31[2, 2]-0.832) < 1e-3
assert abs(1/np.sqrt(2) - 0.707) < 1e-3
# Problem 32: ratio of top discrete eigenvalue to exact tends to 4/pi^2
prev = None
for Nb in [10, 1000, 100000]:
    hb = 2*L/(Nb+1)
    rb = (4/hb**2*np.sin(Nb*np.pi*hb/(4*L))**2)/(Nb*q)**2
    if prev is not None:
        assert abs(rb - 4/np.pi**2) < abs(prev - 4/np.pi**2)
    prev = rb
assert abs(prev - 4/np.pi**2) < 1e-4
assert abs(4/np.pi**2 - 0.405) < 1e-3
# Problem 33
for Nn in [5, 20]:
    for k in range(1, Nn+1):
        assert abs(np.sum(np.sin(np.arange(1, Nn+1)*k*np.pi/(Nn+1))**2) - (Nn+1)/2) < 1e-10

print('all checks passed')
