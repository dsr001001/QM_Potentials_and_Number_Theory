import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(29))
rng = np.random.default_rng(0)
r = np.sort(rng.uniform(t[0], t[-1], 29))
e = np.linspace(t[0], t[-1], 29)
fig, ax = plt.subplots(figsize=(6.5, 3))
for yv, pts, lab in [(2, t, 'zeros'), (1, r, 'random'), (0, e, 'equal spacing')]:
    ax.vlines(pts, yv - 0.35, yv + 0.35, color='k', lw=1)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(['zeros', 'random', 'equally spaced'])
ax.set_xlabel('height $t$ (dimensionless)')
ax.set_xlim(t[0] - 3, t[-1] + 3)
plt.savefig('intro_barcode.pdf', bbox_inches='tight')
