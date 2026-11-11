import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(-L, L, 801)
psi1 = np.sin(1*np.pi*(x + L)/(2*L))/np.sqrt(L)
psi2 = np.sin(2*np.pi*(x + L)/(2*L))/np.sqrt(L)
E1, E2 = (np.pi/(2*L))**2, (2*np.pi/(2*L))**2
T = 2*np.pi/(E2 - E1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for t, ls, lab in [(0, "-", "t = 0"), (T/4, "--", "t = T/4"), (T/2, ":", "t = T/2")]:
    Psi = (psi1*np.exp(-1j*E1*t) + psi2*np.exp(-1j*E2*t))/np.sqrt(2)
    a1.plot(x, np.abs(Psi)**2, ls, label=lab)
a1.set_xlabel(r"position $x/L$")
a1.set_ylabel(r"$|\Psi|^2$ (units of $1/L$)")
a1.legend(fontsize=8)
I = np.trapezoid(x*psi1*psi2, x)
tt = np.linspace(0, 2*T, 400)
a2.plot(tt/T, I*np.cos((E2 - E1)*tt), "k-")
a2.set_xlabel(r"time $t/T$")
a2.set_ylabel(r"$\langle x\rangle / L$")
plt.tight_layout()
plt.savefig("schrodinger_beats.pdf", bbox_inches="tight")
