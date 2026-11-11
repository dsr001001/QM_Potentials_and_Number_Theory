import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 1001)
d = qmnt.fluct(qmnt.zeros(1000), n)
mu, sg = d.mean(), d.std()
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(d, bins=40, density=True, color='0.75', edgecolor='k', label='zeros 1-1000')
x = np.linspace(d.min() - 0.1, d.max() + 0.1, 400)
ax.plot(x, np.exp(-(x - mu)**2/(2*sg**2))/(sg*np.sqrt(2*np.pi)), 'r', lw=2,
        label='Gaussian, same mean and spread')
ax.set_xlabel(r'fluctuation $\delta_n$ (mean spacings)')
ax.set_ylabel('density (per mean spacing)')
ax.legend()
plt.savefig('probability_hist_fluct.pdf', bbox_inches='tight')
