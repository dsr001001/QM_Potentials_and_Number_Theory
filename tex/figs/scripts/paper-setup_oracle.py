import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import lambertw
import qmnt

def smooth_zero(n):
    z = (n - 11/8) / np.e
    return 2*np.pi*(n - 11/8) / np.real(lambertw(z, 0))

n = np.arange(370, 461)
t = np.asarray(qmnt.zeros(460), float)[n - 1]
tp = np.asarray(qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3), float)
s = smooth_zero(n)
g = 2*np.pi / np.log(s / (2*np.pi))
fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(400.5, 460.5, color="0.85")
ax.plot(n, (t - s)/g, "ko", ms=3, label="true fluctuation")
ax.plot(n, (tp - s)/g, "b-", lw=1.2, label="six-prime explicit formula")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("zero index $n$")
ax.set_ylabel("$(t_n - s_n)/\\mathrm{gap}_n$ (mean spacings)")
ax.legend(fontsize=8, loc="lower left")
plt.savefig("paper-setup_oracle.pdf", bbox_inches="tight")
