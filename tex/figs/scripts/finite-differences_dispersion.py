import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5,3))
z = np.linspace(1e-6, 1, 400)
a.plot(z, (np.sin(np.pi*z/2)/(np.pi*z/2))**2, 'k-')
a.set_xlabel(r'$qh/\pi$ (dimensionless)'); a.set_ylabel(r'$K_h(q)/q^2$')
L, N = 44.883, 6000; h = 2*L/(N+1)
k = np.arange(1, 701)
E = (k*np.pi/(2*L))**2
Eh = 4/h**2*np.sin(k*np.pi*h/(4*L))**2
spc = (2*k+1)*(np.pi/(2*L))**2
b.plot(k, (E-Eh)/spc, 'k-', label='exact')
b.plot(k, h**2*E**2/12/spc, 'C1--', label=r'$h^2E_k^2/12$')
b.axvspan(401, 500, color='0.85')
b.set_xlabel('level index k'); b.set_ylabel('deficit (level spacings)')
b.legend(fontsize=8)
plt.tight_layout()
plt.savefig("finite-differences_dispersion.pdf", bbox_inches="tight")
