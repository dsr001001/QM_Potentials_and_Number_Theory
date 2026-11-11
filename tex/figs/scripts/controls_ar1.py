import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(3)
def ar1(phi, n):
    e = rng.normal(size=n + 200)
    x = np.empty(n + 200); x[0] = e[0]
    for j in range(1, n + 200):
        x[j] = phi*x[j-1] + e[j]
    return x[200:]
def fn(phi, N):
    k = np.arange(1, N)
    return 1 + 2*np.sum((1 - k/N)*phi**k)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for off, phi in zip([8, 0, -8], [0.5, 0.0, -0.5]):
    a1.plot(np.arange(100), ar1(phi, 100) + off, lw=1, label='phi = %.1f' % phi)
a1.set_xlabel('index j (dimensionless)')
a1.set_ylabel('X_j + offset (arb. units)')
a1.legend(fontsize=7, loc='upper right')
phi, n = 0.5, 400
g0 = 1/(1 - phi**2)
true_se = np.sqrt(g0*fn(phi, n)/n)
ells = [1, 2, 3, 5, 8, 12, 20, 30, 40]
sim = []
for ell in ells:
    vals = []
    nb = -(-n // ell)
    for _ in range(40):
        x = ar1(phi, n)
        st = rng.integers(0, n, size=(300, nb))
        idx = ((st[:, :, None] + np.arange(ell)[None, None, :]) % n).reshape(300, -1)[:, :n]
        vals.append(x[idx].mean(axis=1).std())
    sim.append(np.mean(vals)/true_se)
pred = [np.sqrt(g0*fn(phi, ell)/n)/true_se for ell in ells]
a2.plot(ells, sim, 'o-', label='simulation')
a2.plot(ells, pred, 'k--', label='block formula, true autocov.')
a2.axhline(1, color='0.6', lw=0.8)
a2.set_xlabel('block length (number of points)')
a2.set_ylabel('bootstrap SE / true SE')
a2.legend(fontsize=7, loc='lower right')
fig.tight_layout()
plt.savefig('controls_ar1.pdf', bbox_inches='tight')
