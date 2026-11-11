import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
t = np.linspace(0, 20, 1000)
A, w, ph = 1.5, 2*np.pi/8, 0.8
ax[0].plot(t, A*np.cos(w*t+ph), 'k')
tm = (2*np.pi - ph)/w
ax[0].annotate('', xy=(tm, A), xytext=(tm, 0), arrowprops=dict(arrowstyle='<->'))
ax[0].text(tm+0.3, 0.6, 'A')
ax[0].annotate('', xy=(tm+8, A+0.2), xytext=(tm, A+0.2), arrowprops=dict(arrowstyle='<->'))
ax[0].text(tm+3, A+0.3, r'$\tau=2\pi/\omega$')
ax[0].set_ylim(-2, 2.1); ax[0].axhline(0, color='gray', lw=0.5)
ax[0].set_xlabel('t (dimensionless)'); ax[0].set_ylabel('f(t)')
t = np.linspace(0, 30, 2000); tot = 0
for m, c in zip([1, 2, 3], ['C0', 'C1', 'C2']):
    y = -np.sin(m*t*np.log(2))/(np.pi*m*2**(m/2)); tot = tot + y
    ax[1].plot(t, y, c, lw=1, label=f'm={m}')
ax[1].plot(t, tot, 'k', lw=1.8, label='sum (RE2)')
for j in range(4):
    ax[1].axvline(j*2*np.pi/np.log(2), ls='--', color='gray', lw=0.7)
ax[1].set_xlabel('t (dimensionless)'); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("fourier-series_sinusoid.pdf", bbox_inches="tight")
