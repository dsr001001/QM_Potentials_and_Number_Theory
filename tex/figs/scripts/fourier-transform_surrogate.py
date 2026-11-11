import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401); d = qmnt.fluct(qmnt.zeros(400), n)
rng = np.random.default_rng(1)
X = np.fft.rfft(d); phi = rng.uniform(0, 2*np.pi, len(X)); phi[0] = 0; phi[-1] = 0
ds = np.fft.irfft(X*np.exp(1j*phi), n=len(d))
def ac(x):
    x = x - x.mean(); r = np.fft.ifft(np.abs(np.fft.fft(x))**2).real; return r/r[0]
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].plot(n[100:200], d[100:200], '-', label="zeros")
ax[0].plot(n[100:200], ds[100:200], '--', label="surrogate")
ax[0].set_xlabel(r"index $n$"); ax[0].set_ylabel(r"$d_n$ (mean spacings)"); ax[0].legend(fontsize=7)
lag = np.arange(11)
ax[1].plot(lag, ac(d)[:11], 'o', mfc='none', label="zeros")
ax[1].plot(lag, ac(ds)[:11], 'x', label="surrogate")
ax[1].axhline(0, color='gray', lw=0.5)
ax[1].set_xlabel("lag $m$"); ax[1].set_ylabel("normalized autocorrelation"); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("fourier-transform_surrogate.pdf", bbox_inches="tight")
