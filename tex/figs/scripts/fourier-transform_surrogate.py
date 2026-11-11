import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401)
d = qmnt.fluct(qmnt.zeros(400), n)
mu = d.mean(); d0 = d - mu
rng = np.random.default_rng(1)
X = np.fft.rfft(d0)
phi = rng.uniform(0, 2*np.pi, len(X)); phi[0] = 0.0; phi[-1] = 0.0
sur = np.fft.irfft(X*np.exp(1j*phi), n=400)
ca = lambda z: np.fft.ifft(np.abs(np.fft.fft(z))**2).real
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].plot(n[100:200], d[100:200], '-', lw=0.9, label='zeros')
ax[0].plot(n[100:200], sur[100:200] + mu, '--', lw=0.9, label='surrogate')
ax[0].set_xlabel('zero index n'); ax[0].set_ylabel('d_n (mean spacings)')
ax[0].legend(fontsize=8)
lags = np.arange(0, 11)
ax[1].plot(lags, (ca(d0)/ca(d0)[0])[lags], 'o', mfc='none', label='zeros')
ax[1].plot(lags, (ca(sur)/ca(sur)[0])[lags], 'x', label='surrogate')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('lag m (levels)'); ax[1].set_ylabel('normalized autocorrelation')
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("fourier-transform_surrogate.pdf", bbox_inches="tight")
