import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def shifts(K, j=1, c=1.0):
    k = np.arange(1, K+1); E = k**2.0
    kk, ll = np.meshgrid(k, k, indexing="ij")
    A = c*(-1)**j/2*((abs(kk-ll)==2*j).astype(float)-(kk+ll==2*j).astype(float))
    D = E[:,None]-E[None,:]; np.fill_diagonal(D, np.inf)
    return k, -(A**2/D).sum(axis=1)
k1, d1 = shifts(400); k2, d2 = shifts(30)
plt.figure(figsize=(6,4))
plt.plot(k1[:30], d1[:30]*k1[:30]**2, "o", mfc="none", label="400 states")
plt.plot(k2, d2*k2**2, "s", ms=4, label="30 states")
plt.xlabel("level index n"); plt.ylabel(r"$n^2\,\delta E_n$ (units of $E_1$)")
plt.legend(); plt.title("Truncating the state sum biases the top levels")
plt.savefig("paper-spurious-floor_truncation.pdf", bbox_inches="tight")
