import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(6.5,3), sharey=True)
ax = axs[0]
ax.plot([0], [0.2321], 'o', color='C0', label='surrogate')
ax.plot([0], [0.2523], 's', color='C1', label='exact')
ax.plot([1], [0.2358], 'o', color='C0')
ax.plot([1], [0.2322], 'o', mfc='none', color='C0', label='before repair')
ax.plot([1], [0.2477], 's', color='C1')
ax.set_title('unconstrained')
axs[1].plot([0, 1], [0.2335, 0.2422], 'o', color='C0')
axs[1].plot([0, 1], [0.2391, 0.2423], 's', color='C1')
axs[1].set_title('fair, $R=4$')
for a in axs:
    a.axhline(0.2438, ls='--', color='k', lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels(['PT-odd', 'abs.-even'])
    a.set_xlim(-0.5, 1.5)
axs[0].set_ylabel('in-sample rms (mean spacings)')
axs[0].legend(fontsize=7, loc='upper left')
plt.savefig('paper-spurious-floor_floors.pdf', bbox_inches='tight')
