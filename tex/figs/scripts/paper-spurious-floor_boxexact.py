import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
K = 80; k = np.arange(1, K+1); E = k**2.0
kk, ll = np.meshgrid(k, k, indexing="ij")
def M(j): return (-1)**j/2*((abs(kk-ll)==2*j).astype(float)-(kk+ll==2*j).astype(float))
c = np.random.default_rng(1).standard_normal(6)
D = E[:,None]-E[None,:]; np.fill_diagonal(D, np.inf)
fig, ax = plt.subplots(1, 2, figsize=(6.5,3))
for a, s in zip(ax, [0.5, 15]):
    A = sum(s*c[j-1]*M(j) for j in range(1, 7))
    d2 = -(A**2/D).sum(axis=1)
    w, v = np.linalg.eig(np.diag(E)+1j*A)
    ex = w[np.argmax(abs(v), axis=1)].real - E
    a.plot(d2[:40], ex[:40], "o", ms=3); lim = max(abs(d2[:40]).max(), abs(ex[:40]).max())
    a.plot([-lim, lim], [-lim, lim], "k--", lw=0.8); a.set_title(f"scale {s}")
    a.set_xlabel("second-order shift"); a.set_ylabel("exact real shift")
plt.tight_layout()
plt.savefig("paper-spurious-floor_boxexact.pdf", bbox_inches="tight")
