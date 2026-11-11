import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ai_zeros
from scipy.linalg import eigh_tridiagonal
a, ap, _, _ = ai_zeros(60)
lin = np.sort(np.concatenate([-a, -ap]))[:100]
n = np.arange(1, 101)
osc = 2.0*n - 1
box = n.astype(float)**2
L, N = 8.0, 8000
hh = 2*L/(N + 1); x = -L + hh*np.arange(1, N + 1)
q4 = eigh_tridiagonal(2/hh**2 + x**4, -np.ones(N - 1)/hh**2, eigvals_only=True,
                      select='i', select_range=(0, 99))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for j, (lev, lab) in enumerate([(lin, '|x|'), (osc, 'x^2'), (box, 'box')]):
    for e in lev[:10]/lev[0]:
        a1.plot([j - 0.3, j + 0.3], [e, e], 'k', lw=1)
a1.set_yscale('log')
a1.set_xticks([0, 1, 2]); a1.set_xticklabels(['$|x|$', '$x^2$', 'box'])
a1.set_ylabel(r'$E_n/E_1$ (dimensionless)')
m = n[:-1]
for lev, lab in [(lin, '$|x|$'), (osc, '$x^2$'), (q4, '$x^4$'), (box, 'box')]:
    sp = np.diff(lev)
    a2.loglog(m, sp/sp[0], label=lab)
a2.set_xlabel(r'level index $n$')
a2.set_ylabel(r'$(E_{n+1}-E_n)/(E_2-E_1)$')
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("bound-states_ladders.pdf", bbox_inches="tight")
