import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
def autocorr(x, kmax):
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x**2)
    return np.array([np.sum(x[:len(x)-k]*x[k:])/den for k in range(kmax + 1)])
n = np.arange(401, 501)
y = qmnt.Nfl(qmnt.smooth_zeros(n), [2], mmax=3)
w = np.random.default_rng(1).normal(size=100)
k = np.arange(31)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, autocorr(y, 30), 'o-', label='RE2 at smooth zeros 401-500')
ax.plot(k, autocorr(w, 30), 's--', label='independent Gaussian noise')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xlabel('lag $k$ (levels)')
ax.set_ylabel('sample autocorrelation $r_k$')
ax.legend()
plt.savefig('probability_autocorr.pdf', bbox_inches='tight')
