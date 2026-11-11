import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t = np.linspace(0, 60, 3000)
L2 = np.log(2.0)
w = -(1 / np.pi) * sum(np.sin(m * t * L2) / (m * 2 ** (m / 2)) for m in (1, 2, 3))
P = 2 * np.pi / L2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, w, 'b-')
for k in (1, 2):
    ax.axvline(k * P, ls=':', color='gray')
ax.annotate('', xy=(P, 0.33), xytext=(0, 0.33), arrowprops=dict(arrowstyle='<->'))
ax.text(P / 2, 0.35, 'period $2\\pi/\\log 2 \\approx 9.065$', ha='center')
ax.set_ylim(-0.4, 0.42)
ax.set_xlabel('height $t$ (dimensionless)')
ax.set_ylabel('prime-2 wave (counting units)')
ax.grid(alpha=0.3)
plt.savefig('intro_prime2.pdf', bbox_inches='tight')
