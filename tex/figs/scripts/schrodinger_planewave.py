import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
k = 2.0
x = np.linspace(0, 2*np.pi, 500)
w = np.exp(1j*k*x)
a1.plot(x, w.real, "-", label="Re")
a1.plot(x, w.imag, "--", label="Im")
a1.plot(x, np.abs(w), ":", color="k", label="|.|")
a1.set_xlabel(r"position $x$ (units of $\ell$)")
a1.set_ylabel(r"$e^{ikx}$, $k=2$")
a1.legend(fontsize=8, loc="lower left")

kk = np.linspace(0, 4, 200)
a2.plot(kk, 2*kk, "-", label=r"string: $\omega = v k$, $v=2$")
a2.plot(kk, kk**2, "--", label=r"matter: $\omega = k^2$")
a2.set_xlabel(r"wavenumber $k$ (units of $1/\ell$)")
a2.set_ylabel(r"$\omega$ (units of $\varepsilon/\hbar$)")
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("schrodinger_planewave.pdf", bbox_inches="tight")
