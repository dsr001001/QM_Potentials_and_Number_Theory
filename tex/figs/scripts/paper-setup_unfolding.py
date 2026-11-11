import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
t = np.asarray(qmnt.zeros(500), float)
def Nsm(x):
    u = x/(2*np.pi); return u*np.log(u) - u + 7/8
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
tt = np.linspace(10, 100, 3000)
N = np.searchsorted(t, tt, side='right')
ax[0].step(tt, N, where='post', color='k', lw=1, label='staircase $N(t)$')
ax[0].plot(tt, Nsm(tt), 'r-', lw=1, label='smooth $N_{sm}(t)$')
ax[0].set_xlabel('height $t$ (dimensionless)'); ax[0].set_ylabel('count')
ax[0].legend(fontsize=7); ax[0].set_title('(a)', fontsize=9)
n = np.arange(1, 501)
f = np.asarray(qmnt.fluct(t, n), float)
ax[1].plot(n, f, '.', ms=2, color='tab:blue')
ax[1].axvspan(400.5, 500.5, color='orange', alpha=0.25)
ax[1].axhline(0, color='0.5', lw=0.5)
ax[1].set_xlabel('zero index $n$'); ax[1].set_ylabel('$f_n$ (mean spacings)')
ax[1].set_title('(b)', fontsize=9)
plt.tight_layout()
plt.savefig('paper-setup_unfolding.pdf', bbox_inches='tight')
