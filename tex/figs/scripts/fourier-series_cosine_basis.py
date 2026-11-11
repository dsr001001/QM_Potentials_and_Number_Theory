import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 44.883
x = np.linspace(-L, L, 8001)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
for k in [0, 1, 2, 5]:
    ax[0].plot(x, np.cos(k*np.pi*x/L), lw=1, label=f'k={k}')
ax[0].set_xlabel('x'); ax[0].set_ylabel(r'$\cos(k\pi x/L)$'); ax[0].legend(fontsize=7)
f = np.exp(-(x-25.5)**2/2) + np.exp(-(x+25.5)**2/2)
ax[1].plot(x, f, 'k', lw=1.5, label='target')
for M in [20, 60, 200]:
    g = np.trapezoid(f, x)/(2*L)*np.ones_like(x)
    for k in range(1, M):
        ck = np.trapezoid(f*np.cos(k*np.pi*x/L), x)/L
        g += ck*np.cos(k*np.pi*x/L)
    ax[1].plot(x, g, lw=0.9, label=f'M={M}')
ax[1].set_xlim(0, L); ax[1].set_xlabel('x'); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("fourier-series_cosine_basis.pdf", bbox_inches="tight")
