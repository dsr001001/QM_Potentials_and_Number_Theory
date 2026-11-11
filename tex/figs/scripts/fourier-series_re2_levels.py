import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
Nsm = lambda t: t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
n = np.arange(300, 501)
s = np.array([brentq(lambda t: Nsm(t)-(k-0.5), 2*np.pi+1e-9, 1e5) for k in n])
m = np.arange(1, 4)
re2 = -np.sum(np.sin(np.outer(s, m*np.log(2)))/(m*2**(m/2)), axis=1)/np.pi
plt.figure(figsize=(6, 4))
plt.axvspan(400.5, 500.5, color='0.9')
plt.plot(n, re2, 'k.-', lw=0.7, ms=3)
plt.xlabel('level index n'); plt.ylabel(r'RE2 at $s_n$ (count units)')
plt.savefig("fourier-series_re2_levels.pdf", bbox_inches="tight")
