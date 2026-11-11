import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(-1, 1, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
for k in range(4):
    ax1.plot(x, np.cos(k*np.pi*x), label="k=%d" % k)
ax1.set_xlabel("x (dimensionless)"); ax1.set_ylabel("cos(k pi x)")
ax1.legend(fontsize=7)
ax2.plot(x, x**2, "k", lw=2, label="x^2")
for K in [1, 2, 3, 5]:
    s = np.full_like(x, 1/3)
    for k in range(1, K):
        s += 4*(-1)**k/(k**2*np.pi**2)*np.cos(k*np.pi*x)
    ax2.plot(x, s, label="K=%d" % K)
ax2.set_xlabel("x (dimensionless)"); ax2.set_ylabel("value (dimensionless)")
ax2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("linear-algebra_cosine.pdf", bbox_inches="tight")
