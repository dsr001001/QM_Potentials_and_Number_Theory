import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
w0 = np.log(2)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
w = np.linspace(w0-1.2, w0+1.2, 3000)
for T, ls in [(20, '-'), (40, '--')]:
    ax[0].plot(w, np.sinc((w-w0)*T/(2*np.pi)), ls, label='T = %d' % T)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('omega (rad per unit t)'); ax[0].set_ylabel('transform / T')
ax[0].legend(fontsize=8); ax[0].set_title('one line at log 2', fontsize=9)
w1, w2 = np.log(7), np.log(8)
w = np.linspace(1.6, 2.45, 3000)
for T, ls in [(30, '-'), (100, '--')]:
    g = np.sinc((w-w1)*T/(2*np.pi)) + np.sinc((w-w2)*T/(2*np.pi))
    ax[1].plot(w, np.abs(g), ls, label='T = %d' % T)
for v in (w1, w2):
    ax[1].axvline(v, color='gray', lw=0.6, ls=':')
ax[1].set_xlabel('omega (rad per unit t)'); ax[1].set_ylabel('|transform| / T')
ax[1].legend(fontsize=8); ax[1].set_title('log 7 and log 8', fontsize=9)
plt.tight_layout()
plt.savefig("fourier-transform_sinc.pdf", bbox_inches="tight")
