import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(370, 461)
t = np.asarray(qmnt.zeros(460))[n - 1]
fl = qmnt.fluct(t, n)
pred = qmnt.oracle_zeros(n, qmnt.first_primes(6))
fp = qmnt.fluct(pred, n)
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(400.5, 460.5, color='0.9', label='held out')
ax.plot(n, fl, 'ko', ms=3, label='true fluctuation')
ax.plot(n, fp, 'b-', label='six-prime formula')
ax.set_xlabel('zero index $n$')
ax.set_ylabel('fluctuation (mean spacings)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(alpha=0.3)
plt.savefig('intro_trainhold.pdf', bbox_inches='tight')
