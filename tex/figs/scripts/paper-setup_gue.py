import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(10000), float)
u = t/(2*np.pi); N = u*np.log(u) - u + 7/8
s = np.diff(N)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(s, bins=60, range=(0, 3), density=True, color='0.8', edgecolor='0.4', label='first $10^4$ zeros')
x = np.linspace(0, 3, 400)
ax.plot(x, 32/np.pi**2*x**2*np.exp(-4*x**2/np.pi), 'b-', label='GUE Wigner surmise')
ax.plot(x, np.exp(-x), 'r--', label='Poisson $e^{-s}$')
ax.set_xlabel('unfolded spacing $s$ (dimensionless)')
ax.set_ylabel('probability density')
ax.legend()
plt.savefig('paper-setup_gue.pdf', bbox_inches='tight')
