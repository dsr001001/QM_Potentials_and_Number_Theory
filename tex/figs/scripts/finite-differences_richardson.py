import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 1.0; E = np.pi**2/4
Np1 = 2**np.arange(1, 11)
h = 2*L/Np1
Eh = 4/h**2*np.sin(np.pi*h/(4*L))**2
raw = E - Eh
rich = np.abs(E - (4*Eh[1:] - Eh[:-1])/3)
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(h, raw, 'o', label='raw grid')
ax.loglog(h[:-1], rich, 's', label='Richardson')
ax.loglog(h, h**2*E**2/12, '--', color='C0')
ax.loglog(h[:-1], h[:-1]**4*E**3/1440, '--', color='C1')
ax.set_xlabel(r'grid spacing $h$ (dimensionless)')
ax.set_ylabel('ground-state energy error')
ax.legend(); ax.grid(True, which='both', alpha=0.3)
plt.savefig("finite-differences_richardson.pdf", bbox_inches="tight")
