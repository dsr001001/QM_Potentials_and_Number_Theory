import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
n = np.arange(1, 501)
rng = np.random.default_rng(1)
rand_te = np.zeros(500, bool); rand_te[rng.choice(400, 80, replace=False)] = True
rows = [
    ('forward', n <= 400, n > 400),
    ('random 80/20', (n <= 400) & ~rand_te, rand_te),
    ('odd/even', (n <= 400) & (n % 2 == 1), (n <= 400) & (n % 2 == 0)),
    ('interpolation', (n < 201) | (n > 300), (n >= 201) & (n <= 300)),
]
fig, ax = plt.subplots(figsize=(6.5, 3))
for i, (name, tr, te) in enumerate(rows):
    y = 3 - i
    ax.vlines(n[tr], y - 0.3, y + 0.3, color='tab:blue', lw=0.5)
    ax.vlines(n[te], y - 0.3, y + 0.3, color='tab:orange', lw=0.5)
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('zero index $n$ (dimensionless)')
ax.set_xlim(0, 501)
ax.plot([], [], color='tab:blue', label='training'); ax.plot([], [], color='tab:orange', label='held out')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)
plt.savefig('paper-setup_splits.pdf', bbox_inches='tight')
