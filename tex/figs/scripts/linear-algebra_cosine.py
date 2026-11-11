import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(-L, L, 500)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
for k in range(4):
    axs[0].plot(x, np.cos(k * np.pi * x / L), label=f"k = {k}")
axs[0].set_xlabel("x (dimensionless, L = 1)"); axs[0].set_ylabel("cos(k pi x / L)")
axs[0].legend(fontsize=7)
axs[1].plot(x, x**2, "k", lw=2, label="x^2")
def partial(K):
    s = np.full_like(x, 1.0 / 3.0)
    for k in range(1, K):
        s += 4 * (-1)**k / (k**2 * np.pi**2) * np.cos(k * np.pi * x)
    return s
for K in [1, 2, 3, 5]:
    axs[1].plot(x, partial(K), lw=1, label=f"K = {K}")
axs[1].set_xlabel("x (dimensionless)"); axs[1].set_ylabel("value")
axs[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("linear-algebra_cosine.pdf", bbox_inches="tight")
