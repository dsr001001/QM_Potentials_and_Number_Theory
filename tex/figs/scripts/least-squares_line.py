import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t = np.array([0, 1, 2, 3.]); y = np.array([1, 3, 4, 4.])
c1, c0 = np.polyfit(t, y, 1)
fig, ax = plt.subplots(figsize=(6, 4))
tt = np.linspace(-0.3, 3.3, 100)
ax.plot(tt, c0 + c1*tt, 'b-', label='least-squares line')
for i, (ti, yi) in enumerate(zip(t, y)):
    ax.plot([ti, ti], [yi, c0 + c1*ti], 'r-', lw=1.5, label='residuals' if i == 0 else None)
ax.plot(t, y, 'ko', label='data')
ax.plot(t.mean(), y.mean(), 'o', mfc='white', mec='b', ms=9, label='point of averages')
ax.set_xlabel('time t (s)'); ax.set_ylabel('position y (m)')
ax.legend(); ax.grid(alpha=0.3)
plt.savefig('least-squares_line.pdf', bbox_inches='tight')
