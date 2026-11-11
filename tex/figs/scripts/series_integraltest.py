import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = lambda x: 1.0 / x
x = np.linspace(1, 8.5, 400)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
ax = axs[0]
k = np.arange(1, 8)
ax.bar(k, f(k), width=1.0, align="edge", color="C1", alpha=0.4, edgecolor="C1")
ax.plot(x, f(x), "k", lw=1.5)
ax.set_title(r"$f(k)$ on $[k,k+1]$: sum $\geq$ integral", fontsize=9)
ax.set_xlabel(r"$x$ (dimensionless)")
ax.set_ylabel(r"$f(x)=1/x$")
ax = axs[1]
k = np.arange(2, 9)
ax.bar(k - 1, f(k), width=1.0, align="edge", color="C0", alpha=0.4, edgecolor="C0")
ax.plot(x, f(x), "k", lw=1.5)
ax.set_title(r"$f(k)$ on $[k-1,k]$: sum $\leq$ integral", fontsize=9)
ax.set_xlabel(r"$x$ (dimensionless)")
for a in axs:
    a.set_xlim(0.8, 8.6); a.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("series_integraltest.pdf", bbox_inches="tight")
