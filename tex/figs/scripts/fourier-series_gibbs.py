import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(-np.pi, np.pi, 4000)
plt.figure(figsize=(6, 4))
plt.plot(x, np.sign(x), 'k', lw=1.5, label='square wave')
for K in [1, 3, 10, 50]:
    S = sum(np.sin((2*j-1)*x)/(2*j-1) for j in range(1, K+1))*4/np.pi
    plt.plot(x, S, lw=0.9, label=f'K={K}')
plt.axhline(1.17898, ls=':', color='k')
plt.xlabel('x (dimensionless)'); plt.ylabel(r'$S_K(x)$'); plt.legend(fontsize=8)
plt.savefig("fourier-series_gibbs.pdf", bbox_inches="tight")
