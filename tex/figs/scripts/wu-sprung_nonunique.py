import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
x = np.linspace(-4.5, 4.5, 800)
V1 = x**2
beta = 0.4
V2 = np.where(x >= 0, x**2/(1+beta)**2, x**2/(1-beta)**2)
for a, V, xp, xm, title in [(ax[0], V1, lambda E: np.sqrt(E), lambda E: -np.sqrt(E), "even: $V=x^2$"),
                            (ax[1], V2, lambda E: (1+beta)*np.sqrt(E), lambda E: -(1-beta)*np.sqrt(E), "sheared, same $w(V)$")]:
    a.plot(x, V, "k")
    for E in [1, 3, 5, 7]:
        a.plot([xm(E), xp(E)], [E, E], "b")
    a.set_ylim(0, 9); a.set_xlim(-4.5, 4.5)
    a.set_xlabel("$x$ (units $\\hbar=2m=1$)"); a.set_title(title, fontsize=10)
ax[0].fill_between([-np.sqrt(4.2), np.sqrt(4.2)], 4.0, 4.2, color="orange", alpha=0.6)
ax[0].set_ylabel("$V$, $E$")
plt.tight_layout()
plt.savefig("wu-sprung_nonunique.pdf", bbox_inches="tight")
