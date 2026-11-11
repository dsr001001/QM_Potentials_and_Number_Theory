import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(40))
t = t[t <= 100]
T = np.linspace(8, 100, 4000)
N = np.searchsorted(t, T, side='right')
u = T / (2 * np.pi)
Ns = u * np.log(u) - u + 7.0 / 8.0
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(T, N, where='post', color='k', label='$N(T)$ (staircase)')
ax.plot(T, Ns, 'r--', label='$N_{\\rm sm}(T)$, paper Eq. (1)')
ax.set_xlabel('height $T$ (dimensionless)')
ax.set_ylabel('number of zeros (dimensionless)')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('intro_staircase.pdf', bbox_inches='tight')
