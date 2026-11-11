import numpy as np
from scipy import integrate
import qmnt


def close(x, y, tol, label):
    assert abs(x - y) < tol, (label, x, y)


rms = lambda e: np.sqrt(np.mean(np.asarray(e) ** 2))

# ---------------- Sample: Train, test and null on four points
t = np.array([1.0, 2.0, 3.0]); y = np.array([1.0, 1.4, 1.2]); y4 = 1.3
m = y.mean(); close(m, 1.2, 1e-12, "S1 mean")
close(rms(y - m), 0.1633, 5e-5, "S1 constant in-sample")
close(abs(y4 - m), 0.1, 1e-12, "S1 constant OOS")
p1 = np.polyfit(t, y, 1)
close(p1[0], 0.1, 1e-9, "S1 slope")
close(rms(y - np.polyval(p1, t)), 0.1414, 5e-5, "S1 line in-sample")
close(np.polyval(p1, 4.0), 1.4, 1e-9, "S1 line prediction")
close(abs(y4 - np.polyval(p1, 4.0)), 0.1, 1e-9, "S1 line OOS")
p2 = np.polyfit(t, y, 2)
close(p2[0], -0.3, 1e-9, "S1 quadratic a")
close(np.max(np.abs(y - np.polyval(p2, t))), 0.0, 1e-9, "S1 quad exact")
close(np.polyval(p2, 4.0), 0.4, 1e-9, "S1 quad prediction")
close(abs(y4 - np.polyval(p2, 4.0)), 0.9, 1e-9, "S1 quad OOS")

# ---------------- Sample: Interpolation works, extrapolation fails (RE4)
tt = np.array([0.0, np.pi / 2, np.pi]); yy = np.array([0.1, 0.9, 0.1])
q = np.polyfit(tt, yy, 2)
close(-q[0], 3.2 / np.pi ** 2, 1e-9, "S2 a")
close(3.2 / np.pi ** 2, 0.324, 5e-4, "S2 a numeric")
close(np.polyval(q, np.pi / 4), 0.7, 1e-9, "S2 q(pi/4)")
close(np.sin(np.pi / 4), 0.7071, 5e-5, "S2 sin(pi/4)")
close(np.sin(np.pi / 4) - 0.7, 0.0071, 5e-5, "S2 interp error")
close(np.polyval(q, 1.5 * np.pi), -2.3, 1e-9, "S2 q(3pi/2)")
close(abs(-1 - np.polyval(q, 1.5 * np.pi)), 1.3, 1e-9, "S2 extrap error")
lin = np.polyfit(tt, yy, 1)
close(lin[0], 0.0, 1e-12, "S2 line slope")
close(np.polyval(lin, 1.5 * np.pi), 0.3667, 5e-5, "S2 line value")
close(abs(-1 - np.polyval(lin, 1.5 * np.pi)), 1.3667, 5e-5, "S2 line error")

# ---------------- Sample: null on RE5 and two predictions
tz = qmnt.zeros(10000)
n = np.arange(401, 501)
d = qmnt.fluct(tz[400:500], n)
close(qmnt.rms(d), 0.2631, 1e-3, "S3 null 401-500")
close(0.369 / 0.263, 1.403, 5e-4, "S3 operator ratio")
close(1 - 0.369 / 0.263, -0.403, 5e-4, "S3 operator skill")
close(0.074 / 0.263, 0.281, 5e-4, "S3 oracle ratio")
close(1 - 0.074 / 0.263, 0.719, 5e-4, "S3 oracle skill")
tp = qmnt.oracle_zeros(n, qmnt.first_primes(6))
r6, c6 = qmnt.score(tp, n)
close(r6, 0.074, 3e-3, "S3 six-prime oracle rms")
close(c6, 0.96, 1.5e-2, "S3 six-prime oracle corr")

# ---------------- Table V nulls (covariate shift listing)
for (a, b), val in zip([(401, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)],
                       [0.263, 0.270, 0.279, 0.288, 0.296]):
    nn = np.arange(a, b + 1)
    close(qmnt.rms(qmnt.fluct(tz[a - 1:b], nn)), val, 2.5e-3, f"null {a}-{b}")

# ---------------- Sample: fluctuations of the right size and no shape
ratio = lambda a, r: np.sqrt(1 + a ** 2 - 2 * r * a)
close(ratio(0.62, 0.0), 1.1766, 5e-5, "S4a ratio")
close(ratio(0.62, 0.0) * 0.263, 0.309, 5e-4, "S4a rms")
close(ratio(0.62, -0.07), 1.2129, 5e-5, "S4b ratio")
close(ratio(0.62, -0.07) * 0.263, 0.319, 5e-4, "S4b rms")
close(ratio(0.16, 0.0), 1.0127, 5e-5, "S4c ratio")
close(ratio(0.16, 0.0) * 0.263, 0.266, 5e-4, "S4c rms")
# checkpoint a=2, rho=0.5
close(ratio(2.0, 0.5), np.sqrt(3), 1e-12, "checkpoint sqrt3")
close(np.sqrt(3), 1.73, 5e-3, "checkpoint 1.73")
# numerical check of the minimum over a
aa = np.linspace(0, 2, 200001)
close(np.min(ratio(aa, 0.96)), np.sqrt(1 - 0.96 ** 2), 1e-6, "min ratio formula")

# ---------------- Sample: counting levels
close(400 / 5, 80, 0, "S5 held out"); close(400 - 80, 320, 0, "S5 trained")
close(len(range(1, 401, 2)), 200, 0, "S5 odd count")
rng = np.random.default_rng(1)
assert len(rng.choice(400, 80, replace=False)) == 80

# ---------------- Sample: infinitely many perfect fits, minimum norm
Xtr = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]); ytr = np.array([1.0, 2.0])
cmn = np.linalg.pinv(Xtr) @ ytr
assert np.allclose(cmn, [1.0, 0.5, 0.5])
close(np.array([1.0, 2.0, 4.0]) @ cmn, 4.0, 1e-12, "S6 prediction")
for c2 in [-3.0, 0.0, 2.5]:
    c = np.array([1.0, 1 - c2, c2]); assert np.allclose(Xtr @ c, ytr)
    close(np.array([1.0, 2.0, 4.0]) @ c, 3 + 2 * c2, 1e-12, "S6 family")

# ---------------- Sample: anticorrelation by construction
r1, r2 = -0.3, -0.1
corr_nbr = r1 * np.sqrt(2 / (1 + r2))
close(corr_nbr, -0.447, 5e-4, "S7 corr")
mse_nbr = (1 + r2) / 2 - 2 * r1 + 1
close(mse_nbr, 2.05, 1e-12, "S7 mse")
close(np.sqrt(mse_nbr), 1.432, 5e-4, "S7 rms ratio")
# simulation check of neighbor-mean formulas with an MA process having these correlations
# (generic check of eq: use a Gaussian sequence with exact Toeplitz covariance)
Cmat = np.array([[1, r1, r2], [r1, 1, r1], [r2, r1, 1]])
w_nb = np.array([0.5, 0, 0.5])
cov_pred_true = w_nb @ Cmat[:, 1]
var_pred = w_nb @ Cmat @ w_nb
close(cov_pred_true / np.sqrt(var_pred), corr_nbr, 1e-12, "S7 matrix check")

# ---------------- Sample: Wigner surmise estimate of rho1
p = lambda s: 32 / np.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / np.pi)
m1 = integrate.quad(lambda s: s * p(s), 0, np.inf)[0]
m2 = integrate.quad(lambda s: s ** 2 * p(s), 0, np.inf)[0]
close(m1, 1.0, 1e-8, "S8 mean 1")
close(m2, 3 * np.pi / 8, 1e-8, "S8 <s^2>")
close(m2, 1.1781, 5e-5, "S8 1.1781")
varu = m2 - 1
close(varu, 0.1781, 5e-5, "S8 var u")
close(2 * 0.263 ** 2, 0.1383, 5e-5, "S8 2 sigma^2")
close(varu / (2 * 0.263 ** 2), 1.287, 5e-4, "S8 ratio")
rho1_est = 1 - varu / (2 * 0.263 ** 2)
close(rho1_est, -0.287, 5e-4, "S8 rho1")

# ---------------- Real zeros: lag-1 negative; neighbor mean anticorrelated (fig + text)
t4 = qmnt.zeros(400); n4 = np.arange(1, 401); d4 = qmnt.fluct(t4, n4)
dc = d4 - d4.mean()
lag1 = np.sum(dc[:-1] * dc[1:]) / np.sum(dc ** 2)
assert lag1 < 0, lag1
ev = np.arange(1, 399, 2)
pred_nb = 0.5 * (d4[ev - 1] + d4[ev + 1])
assert qmnt.corr(pred_nb, d4[ev]) < 0

# kriging listing: sign flip equality (Result)
rng = np.random.default_rng(1)
test = np.zeros(400, bool); test[rng.choice(400, 80, replace=False)] = True
train = ~test
mu = d4[train].mean()
def acf(k):
    i = np.arange(400 - k)
    ok = train[i] & train[i + k]
    a, b = d4[i[ok]] - mu, d4[i[ok] + k] - mu
    return np.mean(a * b) / np.var(d4[train])
w_est = acf(1) / (1 + acf(2))
j = np.array([i for i in range(1, 399) if test[i] and train[i - 1] and train[i + 1]])
s = (d4[j - 1] - mu) + (d4[j + 1] - mu)
cn = qmnt.corr(mu + 0.5 * s, d4[j]); ck = qmnt.corr(mu + w_est * s, d4[j])
if w_est < 0:
    close(ck, -cn, 1e-10, "kriging sign flip")

# ---------------- Sample: kriging versus neighbor mean
w = r1 / (1 + r2)
close(w, -1 / 3, 1e-12, "S9 w")
Emin = 1 - 2 * r1 ** 2 / (1 + r2)
close(Emin, 0.8, 1e-12, "S9 Emin")
close(np.sqrt(Emin), 0.894, 5e-4, "S9 rms ratio")
varp = w ** 2 * 2 * (1 + r2); covp = 2 * w * r1
close(varp, 0.2, 1e-12, "S9 var pred"); close(covp, 0.2, 1e-12, "S9 cov")
close(covp / np.sqrt(varp), 0.447, 5e-4, "S9 corr")
# matrix solve check
Csub = np.array([[1, r2], [r2, 1]]); c0 = np.array([r1, r1])
ww = np.linalg.solve(Csub, c0); assert np.allclose(ww, [w, w])
close(1 - ww @ c0, Emin, 1e-12, "S9 Emin matrix")

# ---------------- Sample: price of a small nugget
wn = r1 / (1 + r2 + 0.05)
close(wn, -0.3158, 5e-5, "S10 w nugget")
close(-4 * wn * r1, -0.3789, 5e-5, "S10 term1")
close(2 * wn ** 2 * (1 + r2), 0.1795, 5e-5, "S10 term2")
En = 1 - 4 * wn * r1 + 2 * wn ** 2 * (1 + r2)
close(En, 0.8006, 5e-5, "S10 E")

# ---------------- Sample: ridge path
x = np.array([1.0, 2.0, 3.0]); yv = np.array([2.1, 3.9, 6.2])
Sxy = np.sum(x * yv); Sxx = np.sum(x * x)
close(Sxy, 28.5, 1e-12, "S11 Sxy"); close(Sxx, 14.0, 1e-12, "S11 Sxx")
cr = lambda mu: Sxy / (Sxx + mu)
close(cr(0), 2.0357, 5e-5, "S11 c0"); close(cr(14), 1.0179, 5e-5, "S11 c14")
close(cr(140), 0.1851, 5e-5, "S11 c140")
close(rms(yv - cr(0) * x), 0.119, 5e-4, "S11 Ein 0")
close(rms(yv - cr(14) * x), 2.20, 5e-3, "S11 Ein 14")
close(4 * cr(0), 8.143, 5e-4, "S11 pred 0")
close(4 * cr(14), 4.071, 5e-4, "S11 pred 14")
close(4 * cr(140), 0.740, 5e-4, "S11 pred 140")
# ridge limit check of SVD filter
A = np.random.default_rng(0).standard_normal((5, 3)); bb = np.ones(5)
U_, S_, Vt_ = np.linalg.svd(A, full_matrices=False)
cbig = Vt_.T @ ((S_ / (S_ ** 2 + 1e12 * S_[0] ** 2)) * (U_.T @ bb))
assert np.max(np.abs(cbig)) < 1e-10

# ---------------- Sample: coverage law
E500 = 811.1843588465; L1 = 44.883
close(np.sqrt(E500), 28.481, 5e-4, "S12 sqrt E")
close(2 * np.sqrt(E500), 56.963, 5e-4, "S12 2 sqrt E")
close(2 * np.sqrt(E500) * L1, 2556.7, 0.1, "S12 product")
close(2 * np.sqrt(E500) * L1 / np.pi, 813.8, 0.1, "S12 Mcrit")
close(300 * np.pi / L1, 21.0, 0.05, "S12 M300")
close(900 * np.pi / L1, 63.0, 0.05, "S12 M900")
close(qmnt.L_of(1000), 44.883, 1e-3, "L scale 1")

# ---------------- Problems
# P2
yp = np.array([3.0, 5.0, 4.0]); close(rms(yp - yp.mean()), 0.816, 5e-4, "P2 rms")
close(abs(6 - yp.mean()), 2.0, 1e-12, "P2 OOS")
# P3
close(0.265 / 0.263, 1.008, 5e-4, "P3")
# P4
dd = np.random.default_rng(5).standard_normal(1000)
close(np.corrcoef(-0.5 * dd, dd)[0, 1], -1.0, 1e-12, "P4 corr")
close(rms(-0.5 * dd - dd) / rms(dd), 1.5, 1e-12, "P4 ratio")
# P5
close(np.sqrt(1 - 0.96 ** 2), 0.280, 5e-4, "P5 min ratio")
close(0.074 / 0.263, 0.281, 5e-4, "P5 oracle ratio")
# P7
th = 0.5; rr1 = -th / (1 + th ** 2)
close(rr1, -0.4, 1e-12, "P7 rho1")
close(rr1 * np.sqrt(2), -0.566, 5e-4, "P7 corr")
close(1 + 0.5 - 2 * rr1, 2.3, 1e-12, "P7 mse")
close(np.sqrt(2.3), 1.517, 5e-4, "P7 rms")
# simulation check of MA(1) autocorrelation
e = np.random.default_rng(7).standard_normal(400001)
dm = e[1:] - th * e[:-1]
dmc = dm - dm.mean()
close(np.sum(dmc[:-1] * dmc[1:]) / np.sum(dmc ** 2), -0.4, 0.01, "P7 sim rho1")
close(np.sum(dmc[:-2] * dmc[2:]) / np.sum(dmc ** 2), 0.0, 0.01, "P7 sim rho2")
# P9
w9 = r1 / (1 + r2 + 0.05); close(w9, -0.3158, 5e-5, "P9 w")
E9 = 1 - 2 * r1 ** 2 / (1 + r2 + 0.05)
close(E9, 0.8105, 5e-5, "P9 E")
# direct check: E = 1 - 4 w r1 + 2 w^2 (1 + r2 + tau)
close(1 - 4 * w9 * r1 + 2 * w9 ** 2 * (1 + r2 + 0.05), E9, 1e-12, "P9 direct")
close(np.sqrt(E9), 0.900, 5e-4, "P9 rms")
# P11
X11 = np.array([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]); y11 = np.array([0.0, 2.0])
c11 = np.linalg.pinv(X11) @ y11
assert np.allclose(c11, [0.5, 1.0, 0.5])
close(np.array([1.0, 2.0, 4.0]) @ c11, 4.5, 1e-12, "P11 pred")
for c2 in [-1.0, 0.0, 3.0]:
    c = np.array([1 - c2, 1.0, c2]); assert np.allclose(X11 @ c, y11)
    close(np.array([1.0, 2.0, 4.0]) @ c, 3 + 3 * c2, 1e-12, "P11 family")
# P13
E1000 = 1419.42248094599; L2 = 59.957
close(np.sqrt(E1000), 37.675, 5e-4, "P13 sqrt")
close(2 * np.sqrt(E1000) * L2 / np.pi, 1438, 1.0, "P13 Mcrit")
close(qmnt.L_of(1500), 59.957, 1e-3, "L scale 2")
# P15
close(0.312 / 0.238, 1.311, 5e-4, "P15 odd"); close(0.311 / 0.250, 1.244, 5e-4, "P15 even")
# P17
close(1 - 0.178 / (2 * 0.25 ** 2), -0.424, 5e-4, "P17")
# P19
close(2.09 / 0.2631, 7.94, 5e-3, "P19")
# P20 (even, quoted)
close(2.6434 / 0.255, 10.37, 5e-3, "P20")
# P23
close(0.1958 / 0.1708, 1.146, 5e-4, "P23")
# P27
close(0.291 / 0.263, 1.106, 5e-4, "P27")
# P29
close(1 - 0.043 / 0.263, 0.837, 5e-4, "P29 oracle25")
close(1 - 0.258 / 0.263, 0.019, 5e-4, "P29 C1")
close(1 - 0.369 / 0.263, -0.403, 5e-4, "P29 operator")

print("all checks passed")
