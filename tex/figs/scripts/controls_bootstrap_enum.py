import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from itertools import product
y = np.array([1., 2., 6.])
means = np.array([y[list(i)].mean() for i in product(range(3), repeat=3)])
vals, counts = np.unique(np.round(means, 10), return_counts=True)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(4/3, 14/3, color='0.88', label='80% percentile interval [4/3, 14/3]')
ax.bar(vals, counts/27, width=0.15, color='tab:blue', zorder=3)
for v, c in zip(vals, counts):
    ax.text(v, c/27 + 0.005, str(c), ha='center', va='bottom', fontsize=9)
ax.axvline(3, ls='--', color='k', label='sample mean = 3')
ax.set_xlabel('mean of the resample (same units as the data)')
ax.set_ylabel('probability')
ax.set_ylim(0, 0.27)
ax.legend(loc='upper right', fontsize=8)
plt.savefig('controls_bootstrap_enum.pdf', bbox_inches='tight')
