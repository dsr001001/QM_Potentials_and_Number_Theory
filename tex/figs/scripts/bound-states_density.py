import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
L = 44.883
E = np.linspace(10, 1500, 500)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(E, L/(np.pi*np.sqrt(E)), label=r'box, $L=44.883$')
ax.semilogy(E, 0.5*np.ones_like(E), label=r'oscillator $x^2$')
ax.semilogy(E, 2/np.pi*np.sqrt(E), label=r'V-shaped $|x|$')
ax.semilogy(E, np.log(E/(2*np.pi))/(2*np.pi), 'k', label='zeros, smooth density')
t = qmnt.zeros(1000)
bins = np.arange(0, 1451, 50)
counts, _ = np.histogram(t, bins=bins)
ax.step(bins[:-1], np.maximum(counts, 1e-3)/50, where='post', color='k', lw=0.8,
        label='first 1000 zeros (bins of 50)')
ax.set_ylim(0.05, 50)
ax.set_xlabel(r'energy $E$ (units $\hbar=2m=1$)')
ax.set_ylabel(r'level density $\rho(E)$ (levels per unit energy)')
ax.legend(fontsize=7)
plt.savefig("bound-states_density.pdf", bbox_inches="tight")
