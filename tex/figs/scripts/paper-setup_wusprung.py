import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
V0 = 3.10073*np.pi
lt = np.log(V0/(2*np.pi*np.e**2))
V = np.linspace(V0, 1600, 4000)
a = np.sqrt(V - V0); b = np.sqrt(V)
x = (a*lt + b*2*np.log((b + a)/np.sqrt(V0)))/np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, V, 'k-', lw=1.2)
ax.plot(-x, V, 'k-', lw=1.2)
for L, c, lab in [(44.883, 'tab:blue', 'scale 1, $L=44.883$'), (59.957, 'tab:red', 'scale 2, $L=59.957$')]:
    ax.axvline(L, ls='--', color=c, label=lab); ax.axvline(-L, ls='--', color=c)
ax.set_xlabel('position $x$ (units $\\hbar=2m=1$)')
ax.set_ylabel('$V(x)$ (units $\\hbar=2m=1$)')
ax.set_ylim(0, 1600)
ax.legend(fontsize=8)
plt.savefig('paper-setup_wusprung.pdf', bbox_inches='tight')
