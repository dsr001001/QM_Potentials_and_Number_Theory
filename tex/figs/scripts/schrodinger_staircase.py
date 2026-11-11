import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import qmnt
    z = np.asarray(qmnt.zeros(40), dtype=float)
except Exception:
    import mpmath
    z = np.array([float(mpmath.zetazero(j).imag) for j in range(1, 41)])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
L = 44.883
Ee = np.linspace(0, 5, 4000)
a1.plot(Ee, np.floor(2*L*np.sqrt(Ee)/np.pi), "k", lw=0.8, label="N_box(E)")
a1.plot(Ee, 2*L*np.sqrt(Ee)/np.pi, "r--", label="2L sqrt(E)/pi")
a1.set_xlabel("E (units hbar = 2m = 1)")
a1.set_ylabel("number of levels")
a1.legend(fontsize=7)
t = np.linspace(1, 100, 4000)
Nt = np.array([np.sum(z <= tt) for tt in t])
Nsm = t/(2*np.pi)*np.log(t/(2*np.pi)) - t/(2*np.pi) + 7/8
a2.plot(t, Nt, "k", lw=0.8, label="N(t)")
a2.plot(t, Nsm, "r--", label="N_sm(t)")
a2.set_xlabel("height t")
a2.set_ylabel("number of zeros")
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("schrodinger_staircase.pdf", bbox_inches="tight")
