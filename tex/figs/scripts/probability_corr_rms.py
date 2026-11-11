import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(401, 501)
y = qmnt.Nfl(qmnt.smooth_zeros(n), [2], mmax=3)
pA = 0.5*y
pB = np.random.default_rng(2).normal(0, y.std(), len(y))
null = np.sqrt(np.mean(y**2))
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for ax, p, ttl in [(axs[0], pA, 'right shape, half amplitude'), (axs[1], pB, 'uncorrelated noise')]:
    r = np.corrcoef(p, y)[0, 1]
    e = np.sqrt(np.mean((p - y)**2))
    ax.plot(n, y, 'k', lw=1.5, label='truth')
    ax.plot(n, p, 'tab:red', lw=1, label='prediction')
    ax.set_title(ttl + '\ncorr %.2f, rms/null %.2f' % (r, e/null), fontsize=9)
    ax.set_xlabel('zero index $n$')
axs[0].set_ylabel('fluctuation (mean spacings)')
axs[0].legend(fontsize=8)
plt.savefig('probability_corr_rms.pdf', bbox_inches='tight')
