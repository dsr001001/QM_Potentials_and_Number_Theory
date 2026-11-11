import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
u = np.linspace(-4, 4, 801)
phi = np.exp(-u**2/2)/np.sqrt(2*np.pi)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(u, phi, 'k', lw=2)
m2 = np.abs(u) <= 2
ax.fill_between(u[m2], phi[m2], color='tab:blue', alpha=0.3, label=r'$|u|<2$: 95.4%')
m1 = np.abs(u) <= 1
ax.fill_between(u[m1], phi[m1], color='tab:blue', alpha=0.7, label=r'$|u|<1$: 68.3%')
ax.set_xlabel(r'z-score $u=(x-\mu)/\sigma$ (dimensionless)')
ax.set_ylabel(r'density $\varphi(u)$')
ax.legend()
plt.savefig('probability_gaussian.pdf', bbox_inches='tight')
