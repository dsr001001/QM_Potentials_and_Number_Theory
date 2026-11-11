import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
xs = np.linspace(0, 1, 400)
for n, ls in zip([1, 2, 3], ["-", "--", ":"]):
    a1.plot(xs, np.sin(n*np.pi*xs), ls, label=f"n = {n}")
a1.axhline(0, color="gray", lw=0.5)
a1.set_xlabel(r"position $x/\ell$")
a1.set_ylabel("displacement (arb. units)")
a1.set_title("string harmonics")
a1.legend(fontsize=8)

L = 1.0
x = np.linspace(-L, L, 400)
for k, ls in zip([1, 2, 3], ["-", "--", ":"]):
    psi = np.sin(k*np.pi*(x + L)/(2*L))/np.sqrt(L)
    a2.plot(x, psi**2, ls, label=f"k = {k}")
a2.set_xlabel(r"position $x/L$")
a2.set_ylabel(r"$|\psi_k|^2$ (units of $1/L$)")
a2.set_title("particle in a box")
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("schrodinger_standing.pdf", bbox_inches="tight")
