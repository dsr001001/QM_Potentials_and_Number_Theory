import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = 10**4
nn = np.arange(1, N + 1)
fig, ax = plt.subplots(figsize=(6, 4))
for seed in (1, 2, 3):
    x = np.random.default_rng(seed).exponential(1.0, N)
    ax.semilogx(nn, np.cumsum(x)/nn, lw=1, label='seed %d' % seed)
ax.semilogx(nn, 1 + 1/np.sqrt(nn), 'k--', lw=1)
ax.semilogx(nn, 1 - 1/np.sqrt(nn), 'k--', lw=1, label=r'$1\pm1/\sqrt{n}$')
ax.set_ylim(0, 2.5)
ax.set_xlabel('number of draws $n$')
ax.set_ylabel(r'running mean $\bar x_n$ (dimensionless)')
ax.legend()
plt.savefig('probability_lln.pdf', bbox_inches='tight')
