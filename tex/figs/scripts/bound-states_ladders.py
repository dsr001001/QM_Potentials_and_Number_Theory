import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ai_zeros
from scipy.linalg import eigh_tridiagonal

a, ap, _, _ = ai_zeros(8)
lin = np.sort(np.concatenate([-a, -ap]))[:12]
osc = 2*np.arange(1, 13) - 1.0
box = np.arange(1, 13)**2*1.0

def fd(Vf, L, N, k):
    h = 2*L/(N + 1); x = -L + h*np.arange(1, N + 1)
    return eigh_tridiagonal(2/h**2 + Vf(x), -np.ones(N - 1)/h**2,
                            eigvals_only=True, select="i", select_range=(0, k - 1))
quart = fd(lambda x: x**4, 7.0, 4000, 12)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for i, (lev, name) in enumerate([(lin, "|x|"), (osc, "x^2"), (box, "box")]):
    for E in lev[:10]/lev[0]:
        a1.plot([i - 0.3, i + 0.3], [E, E], color="C%d" % i, lw=1.2)
a1.set_yscale("log")
a1.set_xticks([0, 1, 2]); a1.set_xticklabels([r"$V=|x|$", r"$V=x^2$", "box"])
a1.set_ylabel(r"$E_n/E_1$ (dimensionless)")
a1.set_title("first ten levels", fontsize=9)
n = np.arange(1, 12)
for lev, name in [(lin, r"$|x|$"), (osc, r"$x^2$"), (quart, r"$x^4$"), (box, "box")]:
    sp = np.diff(lev)[:11]
    a2.loglog(n, sp/sp[0], "o-", ms=3, label=name)
a2.set_xlabel("level index $n$")
a2.set_ylabel(r"$(E_{n+1}-E_n)/(E_2-E_1)$")
a2.set_title("spacing versus $n$", fontsize=9)
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("bound-states_ladders.pdf", bbox_inches="tight")
