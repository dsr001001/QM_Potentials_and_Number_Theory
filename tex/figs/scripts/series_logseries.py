import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(-1, 0.99, 500)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, -np.log(1 - x), 'k', lw=2.5, label=r'$-\log(1-x)$')
for n in [1, 2, 3, 5, 10]:
    ax.plot(x, sum(x**m/m for m in range(1, n + 1)), lw=1, label=f'$n={n}$')
ax.set_ylim(-1, 5)
ax.set_xlabel('$x$ (dimensionless)')
ax.set_ylabel('value (dimensionless)')
ax.legend(fontsize=8, loc='upper left')
plt.savefig("series_logseries.pdf", bbox_inches="tight")
