import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L, N = 1.0, 15
h = 2 * L / (N + 1)
xj = -L + h * np.arange(1, N + 1)
f = lambda x: np.sin(3 * np.pi * (x + L) / (2 * L))
xx = np.linspace(-L, L, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xx, f(xx), "0.6", lw=1, label="f(x)")
ml, sl, bl = ax.stem(xj, f(xj))
plt.setp(sl, color="C0"); plt.setp(ml, color="C0")
plt.setp(bl, color="k", lw=0.5)
ax.axvline(-L, color="k", lw=2); ax.axvline(L, color="k", lw=2)
ax.set_xlabel("x (dimensionless)"); ax.set_ylabel("f (dimensionless)")
ax.set_title("N = 15 interior points, h = 2L/(N+1)")
ax.legend(["f(x)", "stored values f(x_j)"], loc="lower left", fontsize=8)
plt.savefig("linear-algebra_grid.pdf", bbox_inches="tight")
