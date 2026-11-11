import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(370, 461)
t = np.asarray(qmnt.zeros(460), float)[n - 1]
f = np.asarray(qmnt.fluct(t, n), float)
tp = np.asarray(qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3), float)
fp = np.asarray(qmnt.fluct(tp, n), float)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(400.5, 460.5, color='0.85')
ax.plot(n, f, 'ko', ms=3, label='true $f_n$')
ax.plot(n, fp, 'r-', lw=1.2, label='six-prime explicit formula')
ax.axvline(400.5, color='0.4', lw=0.8)
ax.set_xlabel('zero index $n$')
ax.set_ylabel('fluctuation (mean spacings)')
ax.legend(fontsize=8)
plt.savefig('paper-setup_oracle.pdf', bbox_inches='tight')
