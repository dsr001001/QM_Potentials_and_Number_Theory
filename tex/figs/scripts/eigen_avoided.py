import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

g = 0.3
eps = np.linspace(-2, 2, 201)
lo, hi, olo, ohi = [], [], [], []
for ep in eps:
    w, U = np.linalg.eigh(np.array([[ep, g], [g, -ep]]))
    lo.append(w[0]); hi.append(w[1])
    olo.append(abs(U[0, 0])**2); ohi.append(abs(U[0, 1])**2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(eps, eps, "k--", lw=0.7); a1.plot(eps, -eps, "k--", lw=0.7)
sc = a1.scatter(eps, lo, c=olo, cmap="coolwarm", s=4, vmin=0, vmax=1)
a1.scatter(eps, hi, c=ohi, cmap="coolwarm", s=4, vmin=0, vmax=1)
plt.colorbar(sc, ax=a1, label="$|e_1^\\dagger u|^2$")
a1.set_xlabel("$\\varepsilon$ (dimensionless)")
a1.set_ylabel("eigenvalue (dimensionless)")

gs = np.linspace(0, 2, 201)
re1, re2, im1, im2 = [], [], [], []
for gg in gs:
    lam = 2 + np.sqrt(complex(1 - gg**2))*np.array([1, -1])
    re1.append(lam[0].real); re2.append(lam[1].real)
    im1.append(lam[0].imag); im2.append(lam[1].imag)
a2.plot(gs, re1, "b-", label="Re $\\lambda$"); a2.plot(gs, re2, "b-")
a2.plot(gs, im1, "r--", label="Im $\\lambda$"); a2.plot(gs, im2, "r--")
a2.axvline(1, color="gray", lw=0.5)
a2.set_xlabel("$g$ (dimensionless)")
a2.set_ylabel("eigenvalue parts (dimensionless)")
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("eigen_avoided.pdf", bbox_inches="tight")
