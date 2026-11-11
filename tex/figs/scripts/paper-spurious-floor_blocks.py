import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401)
d = qmnt.fluct(qmnt.zeros(400), n)
ws = [3, 5, 7, 9, 13]
b = [np.sqrt(np.mean(np.convolve(d, np.ones(w)/w, 'valid')**2)) for w in ws]
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(ws, b, 'o', label='qmnt, $(t_n-s_n)/\\mathrm{gap}_n$')
ax.plot(ws, [0.108, 0.049, 0.033, 0.035, 0.019], 'x', ms=8, label='paper Sec. V A')
ax.axhline(0.232, ls='--', color='k', label='floor 0.232')
ax.axhline(np.sqrt(np.mean(d**2)), ls=':', color='gray', label='rms of $d_n$')
ax.set_xlabel('window $w$ (levels)')
ax.set_ylabel('block-coherent rms (mean spacings)')
ax.set_ylim(0, 0.3)
ax.legend(fontsize=8)
plt.savefig('paper-spurious-floor_blocks.pdf', bbox_inches='tight')
