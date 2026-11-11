import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tj = 0.7*np.arange(10)
eps = np.array([.15, -.12, .10, -.15, .12, -.10, .14, -.13, .11, -.14])
yj = np.sin(tj) + eps
u = lambda x: x/3.15 - 1
tt = np.linspace(0, 6.3, 800)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tt, np.sin(tt), color='0.6', lw=3, label='sin t')
for d, st in [(1, ':'), (3, '--'), (9, '-')]:
    c = np.linalg.lstsq(np.vander(u(tj), d+1), yj, rcond=None)[0]
    ax.plot(tt, np.vander(u(tt), d+1) @ c, st, label='degree %d' % d)
ax.plot(tj, yj, 'ko', label='noisy data')
ax.set_ylim(-2.5, 2.5); ax.set_xlabel('t (dimensionless)'); ax.set_ylabel('y (dimensionless)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.savefig('least-squares_poly.pdf', bbox_inches='tight')
