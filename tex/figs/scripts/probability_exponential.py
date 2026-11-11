import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
rng = np.random.default_rng(1)
pts = np.sort(rng.uniform(0, 1e4, 10**4))
g_pois = np.diff(pts)
z = qmnt.zeros(10000)
u = qmnt.Nsm(z)
g_zeta = np.diff(u)
g_zeta = g_zeta/g_zeta.mean()
s = np.linspace(0, 4, 400)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for ax, g, ttl in [(axs[0], g_pois, 'independent points'), (axs[1], g_zeta, 'zeta zeros 1-10000')]:
    ax.hist(g, bins=np.linspace(0, 4, 41), density=True, color='0.75', edgecolor='k')
    ax.plot(s, np.exp(-s), 'r', lw=2, label=r'$e^{-s}$')
    ax.set_title(ttl)
    ax.set_xlabel('gap $s$ (mean gaps)')
    ax.legend()
axs[0].set_ylabel('density (per mean gap)')
plt.savefig('probability_exponential.pdf', bbox_inches='tight')
