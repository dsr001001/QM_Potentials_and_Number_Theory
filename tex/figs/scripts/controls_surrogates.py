import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def phase_randomize(d, rng):
    D = np.fft.rfft(d)
    ph = rng.uniform(0, 2*np.pi, len(D)); ph[0] = 0.0
    if len(d) % 2 == 0: ph[-1] = 0.0
    return np.fft.irfft(D * np.exp(1j*ph), n=len(d))

def iaaft(d, rng, iters=200):
    amp = np.abs(np.fft.rfft(d)); srt = np.sort(d); y = rng.permutation(d)
    for _ in range(iters):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j*np.angle(Y)), n=len(d))
        y = srt[np.argsort(np.argsort(y))]
    return y

n = np.arange(1, 401)
s = qmnt.smooth_zeros(n)
d = -qmnt.Nfl(s, [2])
rng = np.random.default_rng(4)
surr = [rng.permutation(d), phase_randomize(d, rng), iaaft(d, rng)]
names = ["permutation", "phase-randomized", "IAAFT"]
fig, ax = plt.subplots(figsize=(6, 4))
m = 120
for k, (y, nm) in enumerate(zip(surr, names)):
    off = -1.2 * (k + 1)
    ax.plot(n[:m], d[:m] + off, color="k", lw=0.6)
    ax.plot(n[:m], y[:m] + off, color="orange", lw=0.9)
    ax.text(m + 2, off, nm, fontsize=8, va="center")
ax.plot(n[:m], d[:m], color="k", lw=1.2)
ax.text(m + 2, 0, "RE2 target", fontsize=8, va="center")
ax.set_xlim(0, m + 40)
ax.set_xlabel("level index n")
ax.set_ylabel("fluctuation (mean spacings, offset)")
plt.savefig("controls_surrogates.pdf", bbox_inches="tight")
