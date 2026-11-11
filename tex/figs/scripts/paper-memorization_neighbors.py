import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 61)
d = qmnt.fluct(qmnt.zeros(60), n)
q = 0.5*(d[:-2] + d[2:])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, d, 'o-', color='C0', ms=4, lw=0.8, label='true $d_n$')
ax.plot(n[1:-1], q, 's', mfc='none', color='C3', ms=5, label='neighbor average')
ax.axhline(0.0, color='k', lw=0.5)
ax.set_xlabel('zero index $n$')
ax.set_ylabel('fluctuation (mean spacings)')
ax.legend(fontsize=8)
plt.savefig('paper-memorization_neighbors.pdf', bbox_inches='tight')
