import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
K = 80
k = np.arange(1, K+1); kk, ll = np.meshgrid(k, k, indexing='ij'); E = k**2.0
def boxA(coeffs):
    A = np.zeros((K, K))
    for j, c in coeffs.items():
        A += c*(-1)**j/2*((abs(kk-ll)==2*j).astype(float)-(kk+ll==2*j).astype(float))
    return A
cr = np.random.default_rng(1).standard_normal(6)
fig, axs = plt.subplots(1, 2, figsize=(6.5,3))
for ax, sc in zip(axs, [0.5, 15]):
    A = boxA({j: sc*cr[j-1] for j in range(1, 7)})
    D = E[:,None]-E[None,:]; np.fill_diagonal(D, np.inf)
    sec = -(A**2/D).sum(axis=1)
    w, v = np.linalg.eig(np.diag(E)+1j*A)
    ex = w[np.argmax(abs(v), axis=1)].real-E
    ax.plot(sec[:40], ex[:40], 'o', ms=3)
    lim = max(abs(sec[:40]).max(), abs(ex[:40]).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlabel('second-order shift (units $E_1$)')
    ax.set_ylabel('exact real shift (units $E_1$)')
    ax.set_title('coefficients x %g' % sc)
plt.tight_layout()
plt.savefig('paper-spurious-floor_boxexact.pdf', bbox_inches='tight')
