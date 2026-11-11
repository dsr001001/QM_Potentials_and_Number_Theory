import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = np.arange(1, 41)
alt = np.cumsum((-1.0)**(n + 1)/n)
terms = []
for j in range(1, 20):
    terms += [1/(2*j - 1), -1/(4*j - 2), -1/(4*j)]
R = np.cumsum(terms)[:40]
L = np.log(2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(n, L - 1/(n + 1), L + 1/(n + 1), color='gray', alpha=0.2, label=r'$\log 2\pm 1/(n+1)$')
ax.plot(n, alt, 'o', ms=4, label='alternating harmonic')
ax.plot(n, R, 's', ms=4, label='rearrangement')
ax.axhline(L, color='C0', lw=0.8)
ax.axhline(L/2, color='C1', lw=0.8, ls='--')
ax.set_xlabel('number of terms $n$ (dimensionless)')
ax.set_ylabel('partial sum (dimensionless)')
ax.legend(fontsize=8)
plt.savefig("series_alternating.pdf", bbox_inches="tight")
