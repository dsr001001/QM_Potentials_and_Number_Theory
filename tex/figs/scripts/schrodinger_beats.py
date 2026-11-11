import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 1.0
x = np.linspace(-L, L, 2001)
psi = lambda k: np.sin(k*np.pi*(x + L)/(2*L))/np.sqrt(L)
E = lambda k: (k*np.pi/(2*L))**2
p1, p2 = psi(1), psi(2)
dE = E(2) - E(1)
T = 2*np.pi/dE
def dens(t):
    Psi = (p1*np.exp(-1j*E(1)*t) + p2*np.exp(-1j*E(2)*t))/np.sqrt(2)
    return np.abs(Psi)**2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for t, lab in [(0, "t=0"), (T/4, "t=T/4"), (T/2, "t=T/2")]:
    a1.plot(x, dens(t), label=lab)
a1.set_xlabel("x / L")
a1.set_ylabel("|Psi|^2  (units 1/L)")
a1.legend(fontsize=7)
ts = np.linspace(0, 2*T, 300)
xav = [np.trapezoid(x*dens(t), x) for t in ts]
a2.plot(ts/T, xav)
a2.axhline(0, color="k", lw=0.5)
a2.set_xlabel("t / T")
a2.set_ylabel("<x> / L")
plt.tight_layout()
plt.savefig("schrodinger_beats.pdf", bbox_inches="tight")
