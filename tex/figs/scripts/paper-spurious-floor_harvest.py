import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(2); n = 400
y = 0.24*rng.standard_normal(n)
x = np.linspace(0, 1, n)
G = np.array([np.cos(k*np.pi*x) for k in range(20)]).T
Z = 1e-7*rng.standard_normal((n, 200))
ks = np.arange(0, 201, 10); rms = []; med = []
for k in ks:
    X = np.hstack([G, Z[:, :k]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms.append(np.sqrt(np.mean((y-X@b)**2)))
    med.append(np.median(abs(b[20:])) if k > 0 else np.nan)
fig, axs = plt.subplots(1, 2, figsize=(6.5,3))
axs[0].plot(ks, rms, 'o-', ms=3)
axs[0].set_xlabel('number of noise columns $k$'); axs[0].set_ylabel('in-sample rms (target units)')
axs[1].semilogy(ks[1:], med[1:], 'o-', ms=3)
axs[1].set_xlabel('number of noise columns $k$'); axs[1].set_ylabel('median |noise coefficient|')
plt.tight_layout()
plt.savefig('paper-spurious-floor_harvest.pdf', bbox_inches='tight')
