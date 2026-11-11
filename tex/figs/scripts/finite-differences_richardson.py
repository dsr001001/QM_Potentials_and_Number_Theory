import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E1 = np.pi**2/4
Np1 = 2.0**np.arange(1, 11)
h = 2/Np1
Er = 4/h**2*np.sin(np.pi*h/4)**2
raw = E1-Er
rich = np.abs(E1-(4*Er[1:]-Er[:-1])/3)
fig, ax = plt.subplots(figsize=(6,4))
ax.loglog(h, raw, 'o', label='raw grid value')
ax.loglog(h[:-1], rich, 's', label='Richardson')
ax.loglog(h, h**2*E1**2/12, 'k--', lw=0.8, label=r'$h^2E^2/12$')
ax.loglog(h, h**4*E1**3/1440, 'k:', lw=0.8, label=r'$h^4E^3/1440$')
ax.set_xlabel('grid spacing h (dimensionless)')
ax.set_ylabel('error in ground-state energy')
ax.legend(fontsize=8)
plt.savefig("finite-differences_richardson.pdf", bbox_inches="tight")
