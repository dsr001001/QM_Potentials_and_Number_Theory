import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
qh = np.linspace(1e-4, np.pi, 400)
a1.plot(qh/np.pi, (np.sin(qh/2)/(qh/2))**2)
a1.set_xlabel(r'$qh/\pi$'); a1.set_ylabel(r'$K_h(q)/q^2$')
a1.grid(alpha=0.3)
L, N = 44.883, 6000
h = 2*L/(N+1)
k = np.arange(1, 801)
Ek = (k*np.pi/(2*L))**2
Eh = 4/h**2*np.sin(k*np.pi*h/(4*L))**2
sp = (2*k+1)*(np.pi/(2*L))**2
a2.plot(k, (Ek-Eh)/sp, '-', label='exact deficit')
a2.plot(k, h**2*Ek**2/12/sp, '--', label=r'$h^2E_k^2/12$')
a2.axvspan(401, 500, alpha=0.2, color='orange')
a2.set_xlabel('level index $k$'); a2.set_ylabel('deficit (level spacings)')
a2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("finite-differences_dispersion.pdf", bbox_inches="tight")
