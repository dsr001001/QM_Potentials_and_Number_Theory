import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(40))
t = t[t <= 100]
rng = np.random.default_rng(0)
r = np.sort(rng.uniform(t[0], t[-1], len(t)))
e = np.linspace(t[0], t[-1], len(t))
fig, ax = plt.subplots(figsize=(6, 4))
for yv, pts in [(2, t), (1, r), (0, e)]:
    ax.vlines(pts, yv - 0.35, yv + 0.35, color='k')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['equally spaced', 'random (seed 0)', 'zeros $t_n$'])
ax.set_xlabel('height $t$ (dimensionless)')
ax.set_ylim(-0.6, 2.6)
plt.savefig('intro_barcode.pdf', bbox_inches='tight')
