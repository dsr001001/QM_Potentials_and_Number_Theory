import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

E = np.linspace(10, 1450, 600)
L = 44.883
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(E, L/(np.pi*np.sqrt(E)), label="box, $L=44.883$")
ax.semilogy(E, 0.5*np.ones_like(E), label=r"oscillator $V=x^2$")
ax.semilogy(E, (2/np.pi)*np.sqrt(E), label=r"V-shaped $V=|x|$")
ax.semilogy(E, np.log(E/(2*np.pi))/(2*np.pi), "k", lw=2,
            label=r"zeros: $\log(t/2\pi)/2\pi$")
t = qmnt.zeros(1000)
bins = np.arange(0, 1450, 50)
cnt, edges = np.histogram(t, bins=bins)
mask = cnt > 0
ax.step(edges[:-1][mask], cnt[mask]/50.0, where="post", color="0.4",
        label="first 1000 zeros (bins of 50)")
ax.set_xlabel(r"energy $E$ or height $t$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"level density $\rho$ (levels per unit energy)")
ax.legend(fontsize=8)
plt.savefig("bound-states_density.pdf", bbox_inches="tight")
