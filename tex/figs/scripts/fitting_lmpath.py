import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3))
s = np.logspace(-4, 0, 400)
for lam in [1e-6, 1e-4, 1e-2, 1.0]:
    a.loglog(s, s/(s**2 + lam), label="lambda = %g" % lam)
a.loglog(s, 1/s, 'k--', label="pseudo-inverse")
a.set_xlabel("sigma / sigma_1 (dimensionless)")
a.set_ylabel("sigma_1 f(sigma) (dimensionless)")
a.legend(fontsize=6); a.set_title("(a)", fontsize=9)
A = np.array([[1.0, 0.9], [0.9, 1.0], [0.3, -0.2]])
r = np.array([1.0, 0.5, 0.2])
gn = np.linalg.lstsq(A, r, rcond=None)[0]
lo = min(0, gn[0], gn[1]) - 0.6; hi = max(0, gn[0], gn[1]) + 0.6
d1, d2 = np.meshgrid(np.linspace(lo, hi, 300), np.linspace(lo, hi, 300))
Q = np.zeros_like(d1)
for i in range(3):
    Q += 0.5*(r[i] - A[i, 0]*d1 - A[i, 1]*d2)**2
b.contour(d1, d2, Q, levels=15, colors='gray', linewidths=0.6)
lams = np.logspace(2, -4, 300)
path = np.array([np.linalg.solve(A.T @ A + l*np.eye(2), A.T @ r) for l in lams])
b.plot(path[:, 0], path[:, 1], 'b-', lw=1.5, label="LM step")
g = A.T @ r; g = 0.5*g/np.linalg.norm(g)
b.annotate("", xy=(g[0], g[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r'))
b.plot(gn[0], gn[1], 'k*', ms=10, label="Gauss-Newton")
b.set_xlabel("delta c_1 (dimensionless)"); b.set_ylabel("delta c_2 (dimensionless)")
b.legend(fontsize=7); b.set_title("(b)", fontsize=9); b.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("fitting_lmpath.pdf", bbox_inches="tight")
