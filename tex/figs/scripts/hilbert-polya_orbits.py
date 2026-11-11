import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
r = np.arange(1, 6)
a1.stem(2*np.pi*r, 1.0/r, basefmt=" ")
a1.set_xlabel(r"traversal time $2\pi r$ (dimensionless)")
a1.set_ylabel("amplitude")
a1.set_title("equally spaced levels", fontsize=9)
pos, h_ef, h_hyp = [], [], []
for p in [2, 3, 5, 7, 11, 13]:
    for m in range(1, 4):
        pos.append(m*np.log(p))
        h_ef.append(1.0/(m*p**(m/2)))
        h_hyp.append(1.0/(m*(p**(m/2) - p**(-m/2))))
a2.stem(pos, h_ef, basefmt=" ", label=r"$1/(m p^{m/2})$")
a2.plot(pos, h_hyp, "o", mfc="none", color="C3", ms=5,
        label=r"$1/(m(p^{m/2}-p^{-m/2}))$")
a2.set_xlabel(r"$m\log p$ (dimensionless)")
a2.set_title("explicit formula, $p\\leq 13$, $m\\leq 3$", fontsize=9)
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("hilbert-polya_orbits.pdf", bbox_inches="tight")
