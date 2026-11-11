import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401)
s = qmnt.smooth_zeros(n)
d = -np.asarray(qmnt.Nfl(s, np.array([2]), mmax=3))
rng = np.random.default_rng(4)
def phase_randomize(d):
    D = np.fft.rfft(d)
    ph = rng.uniform(0, 2*np.pi, len(D)); ph[0] = 0.0
    if len(d) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(D*np.exp(1j*ph), n=len(d))
def iaaft(d, iters=200):
    amp = np.abs(np.fft.rfft(d)); srt = np.sort(d); y = rng.permutation(d)
    for _ in range(iters):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp*np.exp(1j*np.angle(Y)), n=len(d))
        y = srt[np.argsort(np.argsort(y))]
    return y
surr = [rng.permutation(d), phase_randomize(d), iaaft(d)]
names = ['permutation', 'phase-randomized', 'IAAFT']
fig, ax = plt.subplots(figsize=(6, 4))
k = n[:120]
ax.plot(k, d[:120], color='k', lw=1.2, label='prime-2 target')
for i, (sv, nm) in enumerate(zip(surr, names)):
    off = -1.2*(i + 1)
    ax.plot(k, d[:120] + off, color='k', lw=0.5, alpha=0.4)
    ax.plot(k, sv[:120] + off, color='tab:orange', lw=1)
    ax.text(122, off, nm, fontsize=8, va='center')
ax.set_xlim(0, 150)
ax.set_xlabel('level index n')
ax.set_ylabel('fluctuation (mean spacings) + offset')
ax.legend(fontsize=8, loc='upper right')
plt.savefig('controls_surrogates.pdf', bbox_inches='tight')
