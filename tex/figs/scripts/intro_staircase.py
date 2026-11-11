import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(40))
T = np.linspace(8, 100, 5000)
N = np.searchsorted(t, T, side='right')
u = T / (2 * np.pi)
Ns = u * np.log(u) - u + 7.0 / 8.0
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(T, N, drawstyle='steps-post', color='k', label='staircase $N(T)$')
ax.plot(T, Ns, color='tab:red', label='smooth curve $N_{\\rm sm}(T)$')
ax.set_xlabel('height $T$ (dimensionless)')
ax.set_ylabel('number of zeros')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.savefig('intro_staircase.pdf', bbox_inches='tight')
