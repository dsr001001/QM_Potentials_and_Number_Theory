import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t = np.linspace(0, 60, 4000)
L2 = np.log(2.0)
w = -(1/np.pi) * sum(np.sin(m*t*L2) / (m * 2**(m/2)) for m in (1, 2, 3))
P = 2*np.pi / L2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, w, color='k')
for k in (1, 2, 3):
    ax.axvline(k*P, color='tab:blue', ls='--', lw=0.8)
ax.annotate('', xy=(2*P, 0.3), xytext=(P, 0.3), arrowprops=dict(arrowstyle='<->'))
ax.text(1.5*P, 0.32, 'period 9.065', ha='center')
ax.set_xlabel('height $t$ (dimensionless)')
ax.set_ylabel('prime-2 wave (zero counts)')
ax.set_ylim(-0.4, 0.4)
ax.grid(alpha=0.3)
plt.savefig('intro_prime2.pdf', bbox_inches='tight')
