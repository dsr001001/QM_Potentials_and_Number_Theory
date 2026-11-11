import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
ell = 1.0
x = np.linspace(0, ell, 400)
for n in (1, 2, 3):
    a1.plot(x, np.sin(n*np.pi*x/ell), label=f"n={n}")
a1.axhline(0, color="k", lw=0.5)
a1.set_xlabel("x / l (string length l)")
a1.set_ylabel("y / y_max")
a1.set_title("string harmonics")
a1.legend(fontsize=7)
L = 1.0
X = np.linspace(-L, L, 400)
for k in (1, 2, 3):
    a2.plot(X, np.sin(k*np.pi*(X + L)/(2*L))**2/L, label=f"k={k}")
a2.set_xlabel("x / L")
a2.set_ylabel("|psi_k|^2  (units 1/L)")
a2.set_title("box probability densities")
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("schrodinger_standing.pdf", bbox_inches="tight")
