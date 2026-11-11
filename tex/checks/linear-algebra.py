import numpy as np
from scipy.integrate import quad
import qmnt

def close(a, b, tol):
    assert abs(a - b) < tol, (a, b)

# Section 1 checkpoint: p - d
p = np.array([1.0, 2.0, 3.0]); d = np.array([1.5, 1.5, 3.5])
assert np.allclose(p - d, [-0.5, 0.5, -0.5])

# Sample: norm and rms of a residual vector
r = np.array([0.3, -0.1, 0.2, -0.4])
close(r @ r, 0.30, 1e-12)
close(np.linalg.norm(r), 0.5477, 1e-4)
close(np.sqrt(np.mean(r**2)), 0.2739, 1e-4)

# Checkpoint orthogonality
assert np.dot([1, 2, 3], [3, 0, -1]) == 0 and np.dot([1, 1, 1, 1], [1, -1, 1, -1]) == 0

# Sample: right shape wrong size, etc.
d = np.array([1., -1, 2, -2]); p = 0.5 * d; q = np.array([2., -2, -1, 1])
rmsf = lambda v: np.sqrt(np.mean(v**2))
close(rmsf(d), 1.5811, 1e-4)
close(np.corrcoef(p, d)[0, 1], 1.0, 1e-12)
close(rmsf(p - d), 0.7906, 1e-4)
close(q @ d, 0.0, 1e-12); close(q @ q, 10.0, 1e-12)
close(rmsf(q - d), 2.2361, 1e-4)
close(rmsf(q - d) / rmsf(d), np.sqrt(2), 1e-12)

# Checkpoint (1,i).(1,i) without conjugate
u = np.array([1, 1j]); close(abs(u @ u), 0.0, 1e-15)

# Sample: complex inner product
u = np.array([1 + 1j, 2]); v = np.array([1j, 1 - 1j])
assert np.isclose(np.vdot(u, v), 3 - 1j)
assert np.isclose(np.vdot(v, u), 3 + 1j)
close(np.linalg.norm(u)**2, 6, 1e-12); close(np.linalg.norm(v)**2, 3, 1e-12)
close(abs(np.vdot(u, v))**2, 10, 1e-12)

# Sample: components in rotated basis
e1 = np.array([1, 1]) / np.sqrt(2); e2 = np.array([1, -1]) / np.sqrt(2); vv = np.array([3., 1])
a1, a2 = e1 @ vv, e2 @ vv
close(a1, 2.8284, 1e-4); close(a2, 1.4142, 1e-4)
assert np.allclose(a1 * e1 + a2 * e2, vv); close(a1**2 + a2**2, 10, 1e-12)

# Sample: projection onto a line
a = np.array([3., 1]); b = np.array([1., 2])
pr = (a @ b) / (a @ a) * a
assert np.allclose(pr, [1.5, 0.5]); assert np.allclose(b - pr, [-0.5, 1.5])
close(a @ (b - pr), 0, 1e-12)

# Column picture example
assert np.allclose(np.array([[1, 2], [3, 4]]) @ [1, 1], [3, 7])

# Sample: spacing matrix on RE1
t = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
              37.586178, 40.918719, 43.327073, 48.005151, 49.773832])
D = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
g = D @ t[:4]
assert np.allclose(g, [6.887315, 3.988818, 5.414018], atol=1e-9)
close(np.diff(t).sum(), 35.639107, 1e-9)

# Sample rotate/stretch
R = np.array([[0, -1], [1, 0]]); S = np.array([[2, 0], [0, 1]])
assert np.array_equal(R @ S, [[0, -1], [2, 0]]); assert np.array_equal(S @ R, [[0, -2], [1, 0]])
assert np.array_equal(R @ S @ [1, 0], [0, 2]); assert np.array_equal(S @ R @ [1, 0], [0, 1])

# Quadratic form example
c = np.array([1., 2]); T = np.array([[2., 1], [1, 3]])
assert np.allclose(T @ c, [4, 7]); close(c @ T @ c, 18, 1e-12)

# Sample: 2x2 inverse and solve
A = np.array([[4., 7], [2, 6]])
close(np.linalg.det(A), 10, 1e-12)
assert np.allclose(np.linalg.inv(A), [[0.6, -0.7], [-0.2, 0.4]])
assert np.allclose(np.linalg.solve(A, [1, 2]), [-0.8, 0.6])

# Sample: elimination 3x3, determinant, operation count n=3
A3 = np.array([[1., 1, 1], [2, 3, 1], [1, 2, 3]]); b3 = np.array([6., 11, 14])
assert np.allclose(np.linalg.solve(A3, b3), [1, 2, 3])
close(np.linalg.det(A3), 3, 1e-12)
U = A3.copy(); U[1] -= 2 * U[0]; U[2] -= U[0]; U[2] -= U[1]
assert np.allclose(U, [[1, 1, 1], [0, 1, -1], [0, 0, 3]])
cost = lambda n: (n - 1) * n * (n + 1) / 3
assert cost(3) == 8 and sum(j * (j + 1) for j in range(1, 3)) == 8
assert sum(j * (j + 1) for j in range(1, 50)) == cost(50)
close(cost(6000), 7.2e10, 1e7)                     # Problem 27 / text

# det = r1.(r2 x r3) instance
Ar = np.array([[1., 4, -2], [3, 0.5, 2], [-1, 2, 5]])
close(np.linalg.det(Ar), Ar[0] @ np.cross(Ar[1], Ar[2]), 1e-10)

# Null-space special solutions instance (Problem 12 matrix)
A12 = np.array([[1., 2, 3], [2, 4, 6], [1, 0, 1]])
assert np.linalg.matrix_rank(A12) == 2
assert np.allclose(A12 @ [-1, -1, 1], 0)

# Sample: recovering zeros from spacings
gs = np.array([6.887315, 3.988818, 5.414018])
close(gs.sum(), 16.290151, 1e-9)
s = 30.424876
tt = np.array([s - gs.sum(), s - gs[1] - gs[2], s - gs[2], s])
assert np.allclose(D @ tt, gs); close(tt[0], 14.134725, 1e-9)
assert np.allclose(D @ np.ones(4), 0)

# Determinant area figure
close(np.linalg.det(np.array([[2., 1], [1, 2]])), 3, 1e-12)

# Sample: grid sum exact for cos(pi x/2)
for N in [3, 7, 50, 999]:
    h = 2 / (N + 1); x = -1 + h * np.arange(1, N + 1)
    close(h * np.sum(np.cos(np.pi * x / 2)**2), 1.0, 1e-12)
close(quad(lambda x: np.cos(np.pi * x / 2)**2, -1, 1)[0], 1.0, 1e-12)

# Pitfall: interior sum of x^2, and trapezoid values
for N, claimed, trap in [(9, 0.48, 0.6800), (99, 0.6468, 0.6668)]:
    h = 2 / (N + 1); x = -1 + h * np.arange(1, N + 1)
    s_int = h * np.sum(x**2)
    close(s_int, claimed, 1e-4)
    s_tr = s_int + h / 2 * (1 + 1)
    close(s_tr, trap, 1e-4)
close(0.68 - 2 / 3, 0.0133, 1e-4)
close(abs(0.66680 - 2 / 3), 0.0001, 1e-4)
close(2 / 3 - 0.48, 0.19, 5e-3)
close(2 / 3 - 0.6468, 0.02, 1e-3)

# Cosine orthogonality (eq cosorth), numerically with L = 2.5
Lt = 2.5
for j in range(4):
    for k in range(4):
        val = quad(lambda x: np.cos(j * np.pi * x / Lt) * np.cos(k * np.pi * x / Lt), -Lt, Lt)[0]
        exp = 0 if j != k else (2 * Lt if j == 0 else Lt)
        close(val, exp, 1e-9)

# Sample: cosine coefficients of x^2
a0 = 0.5 * quad(lambda x: x**2, -1, 1)[0]; close(a0, 1 / 3, 1e-12)
ak = [quad(lambda x: x**2 * np.cos(k * np.pi * x), -1, 1)[0] for k in range(1, 4)]
for k in range(1, 4):
    close(ak[k - 1], 4 * (-1)**k / (k**2 * np.pi**2), 1e-10)
close(ak[0], -0.40528, 1e-5); close(ak[1], 0.10132, 1e-5); close(ak[2], -0.04503, 1e-5)
ps = np.cumsum([a0] + ak)
close(ps[1], -0.0720, 1e-4); close(ps[2], 0.0294, 1e-4); close(ps[3], -0.0157, 1e-4)

# Box sensitivity matrix: grid vs analytic (sample, listing, Problem 23)
L, N = 44.883, 3000
h = 2 * L / (N + 1); x = -L + h * np.arange(1, N + 1)
nlev, M = 6, 6
n = np.arange(1, nlev + 1); k = np.arange(M)
psi = np.sin(np.outer(x + L, n) * np.pi / (2 * L)) / np.sqrt(L)
Phi = np.cos(np.outer(x, k) * np.pi / L)
Agrid = h * (psi**2).T @ Phi
Aan = np.zeros((nlev, M)); Aan[:, 0] = 1
for nn in range(1, nlev + 1):
    if nn < M:
        Aan[nn - 1, nn] = (-1)**(nn + 1) / 2
assert np.abs(Agrid - Aan).max() < 1e-9
# row n = 6 >= M = 6 has only the 1 in column 0
assert np.allclose(Agrid[5], [1, 0, 0, 0, 0, 0], atol=1e-9)
# independence of L
L2 = 3.0; h2 = 2 * L2 / (N + 1); x2 = -L2 + h2 * np.arange(1, N + 1)
psi2 = np.sin(np.outer(x2 + L2, n) * np.pi / (2 * L2)) / np.sqrt(L2)
A2 = h2 * (psi2**2).T @ np.cos(np.outer(x2, k) * np.pi / L2)
assert np.abs(A2 - Aan).max() < 1e-9

# Sample: too many / too few coefficients in the box
Ab4 = np.array([[1, 0.5, 0, 0], [1, 0, -0.5, 0], [1, 0, 0, 0.5]])
rr = np.array([0.2, -0.1, 0.3])
for s in [-1.0, 0.0, 0.37, 2.0]:
    cc = np.array([0, 0.4, 0.2, 0.6]) + s * np.array([1, -2, 2, -2])
    assert np.allclose(Ab4 @ cc, rr)
assert np.allclose(Ab4 @ [1, -2, 2, -2], 0)
Ab2 = np.array([[1, 0.5], [1, 0], [1, 0]])
assert np.linalg.matrix_rank(Ab4) == 3 and np.linalg.matrix_rank(Ab2) == 2   # Problem 24
aug = np.column_stack([Ab2, rr]); assert np.linalg.matrix_rank(aug) == 3     # inconsistent
close((rr[2] - rr[1]), 0.4, 1e-12)
nl = 7
Ab = np.zeros((nl, nl + 1)); Ab[:, 0] = 1
for nn in range(1, nl + 1):
    Ab[nn - 1, nn] = (-1)**(nn + 1) / 2
assert np.linalg.matrix_rank(Ab) == nl

# Paper connection numbers
close(0.2631 * np.sqrt(100), 2.631, 1e-9)
close(0.263 * np.sqrt(1 - 0.96**2), 0.074, 6e-4)
close(2 * 44.8827 / 6001, 0.01496, 1e-5)          # text and Problem 18
close(2 * 59.9572 / 8001, 0.014987, 1e-6)          # Problem 18
close(2 * 44.8827 / 6001, 0.014958, 1e-6)

# Problem 1
u = np.array([2., -1, 3, 0]); v = np.array([1., 4, 0, -2])
assert np.allclose(u + v, [3, 3, 3, -2]); assert np.allclose(3 * u - 2 * v, [4, -11, 9, 4])
close(u @ v, -2, 1e-12); close(np.linalg.norm(u), 3.7417, 1e-4); close(np.linalg.norm(v), 4.5826, 1e-4)
# Problem 2
w = np.array([1., 2, 2]); assert np.allclose(w / np.linalg.norm(w), [1 / 3, 2 / 3, 2 / 3])
# Problem 3
close(rmsf(np.array([0.1, -0.2, 0.2, 0.0, -0.1])), 0.1414, 1e-4)
# Problem 4
u = np.array([1j, 1 + 2j]); w = np.array([1, 1j])
close(np.linalg.norm(u)**2, 6, 1e-12); close(np.linalg.norm(w)**2, 2, 1e-12)
assert np.isclose(np.vdot(u, w), 2)
# Problem 6
A6 = np.array([[2, 1], [0, 3], [1, -1]]); assert np.array_equal(A6 @ [1, 2], [4, 6, -1])
# Problem 7
A7 = np.array([[1, 2], [3, 4]]); B7 = np.array([[0, 1], [1, 0]])
assert np.array_equal(A7 @ B7, [[2, 1], [4, 3]]); assert np.array_equal(B7 @ A7, [[3, 4], [1, 2]])
# Problem 9
A9 = np.array([[3., 1], [5, 2]]); assert np.allclose(np.linalg.inv(A9), [[2, -1], [-5, 3]])
# Problem 10
x10 = np.linalg.solve(np.array([[2., 1, -1], [1, 3, 2], [3, -1, 1]]), [1, 13, 4])
assert np.allclose(np.array([[2., 1, -1], [1, 3, 2], [3, -1, 1]]) @ x10, [1, 13, 4])
# Problem 11
assert np.linalg.matrix_rank(np.array([[1, 2, 3], [2, 4, 6]])) == 1
assert np.linalg.matrix_rank(np.array([[1, 2, 3], [2, 4, 7]])) == 2
# Problem 13
A13 = np.array([[2., 0, 1], [1, 3, 2], [1, 1, 1]])
close(np.linalg.det(A13), 0, 1e-12)
assert np.allclose(A13[:, 2], 0.5 * A13[:, 0] + 0.5 * A13[:, 1])
# Problem 14
close(abs(np.linalg.det(np.array([[3., 1], [1, 2]]))), 5, 1e-12)
# Problem 15
Lp = 1.7
close(quad(lambda x: np.cos(3 * np.pi * x / Lp)**2, -Lp, Lp)[0], Lp, 1e-9)
close(quad(lambda x: np.cos(np.pi * x / Lp) * np.cos(2 * np.pi * x / Lp), -Lp, Lp)[0], 0, 1e-9)
close(quad(lambda x: np.sin(np.pi * x / Lp) * np.cos(np.pi * x / Lp), -Lp, Lp)[0], 0, 1e-9)
# Problem 17
G = np.array([[1, 0.5], [0.5, 1 / 3]]); bb = np.array([1 / 3, 0.25])
assert np.allclose(np.linalg.solve(G, bb), [-1 / 6, 1])
# Problem 19 shapes: null dims
assert (400 - 300, 900 - 800, 400 - 200) == (100, 100, 200)
# Problem 25: smooth-zero null rms on 401-500
nn = np.arange(401, 501)
dfl = qmnt.fluct(qmnt.zeros(500)[400:], nn)
close(qmnt.rms(dfl), 0.2631, 2e-3)
# Problem 29
th = 0.7; Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
xv = np.array([1.3, -0.4]); close(np.linalg.norm(Rt @ xv), np.linalg.norm(xv), 1e-12)
close(np.linalg.det(Rt), 1, 1e-12)
# Problem 31
s0 = 3.2 / 26; close(s0, 8 / 65, 1e-12); close(s0, 0.1231, 1e-4)
cmin = np.array([0, 0.4, 0.2, 0.6]) + s0 * np.array([1, -2, 2, -2])
assert np.allclose(cmin, [0.1231, 0.1538, 0.4462, 0.3538], atol=1e-4)
close(cmin @ np.array([1, -2, 2, -2]), 0, 1e-12)
assert np.allclose(cmin, np.linalg.pinv(Ab4) @ rr)

print("All checks passed.")
