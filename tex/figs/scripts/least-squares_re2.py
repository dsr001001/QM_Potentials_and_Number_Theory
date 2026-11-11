import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
L2 = np.log(2)
F = lambda x: -sum(np.sin(m*x*L2)/(m*2**(m/2)) for m in (1, 2, 3))/np.pi
des = lambda x, w: np.hstack([np.sin(np.outer(x, w)), np.cos(np.outer(x, w))])
ts = np.linspace(10, 60, 80)
w3 = L2*np.array([1, 2, 3.]); wg = 8*np.arange(1, 61)/60
c3 = np.linalg.lstsq(des(ts, w3), F(ts), rcond=None)[0]
cg = np.linalg.lstsq(des(ts, wg), F(ts), rcond=None)[0]
tt = np.linspace(0, 100, 3000)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(10, 60, color='0.9', label='training window')
ax.plot(tt, F(tt), color='0.5', lw=3, label='true wave')
ax.plot(tt, des(tt, w3) @ c3, 'b--', label='3 correct frequencies')
ax.plot(tt, des(tt, wg) @ cg, 'r-', lw=1, label='120-coefficient grid')
ax.plot(ts, F(ts), 'k.', ms=3)
ax.set_ylim(-1.5, 1.5); ax.set_xlabel('t (dimensionless)'); ax.set_ylabel('F (dimensionless)')
ax.legend(fontsize=7, loc='lower left')
plt.savefig('least-squares_re2.pdf', bbox_inches='tight')
