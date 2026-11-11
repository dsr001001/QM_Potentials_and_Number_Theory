import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import qmnt
    tz = np.asarray(qmnt.zeros(35))
except Exception:
    import mpmath
    tz = np.array([float(mpmath.zetazero(n).imag) for n in range(1, 36)])

def Nsm(t):
    return t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
L = 44.883
E = np.linspace(1e-6, 0.3, 3000)
a1.plot(E, np.floor(2*L*np.sqrt(E)/np.pi), "k-", lw=1, label="staircase")
a1.plot(E, 2*L*np.sqrt(E)/np.pi, "r--", label=r"$2L\sqrt{E}/\pi$")
a1.set_xlabel(r"energy $E$ (units $\hbar=2m=1$)")
a1.set_ylabel(r"$N(E)$")
a1.set_title("empty box, L = 44.883", fontsize=9)
a1.legend(fontsize=8)

t = np.linspace(10, 100, 4000)
Nt = np.array([np.sum(tz <= ti) for ti in t])
a2.plot(t, Nt, "k-", lw=1, label="N(t), zeros")
a2.plot(t, Nsm(t), "r--", label=r"$N_{\rm sm}(t)$")
a2.set_xlabel(r"height $t$")
a2.set_ylabel(r"$N(t)$")
a2.set_title("zeta zeros", fontsize=9)
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("schrodinger_staircase.pdf", bbox_inches="tight")
