import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = np.array([[3, 0], [4, 5.]]); U, s, Vh = np.linalg.svd(A)
th = np.linspace(0, 2*np.pi, 400); C = np.vstack([np.cos(th), np.sin(th)]); E = A @ C
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3))
a.plot(C[0], C[1], 'k-')
for j, col in [(0, 'b'), (1, 'r')]:
    a.annotate('', xy=Vh[j], xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=col, lw=2))
    a.text(1.1*Vh[j][0], 1.1*Vh[j][1], 'v%d' % (j+1), color=col)
    w = s[j]*U[:, j]
    b.annotate('', xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=col, lw=2))
    b.text(1.08*w[0], 1.08*w[1], 'sigma%d u%d' % (j+1, j+1), color=col)
b.plot(E[0], E[1], 'k-')
for ax, ttl in [(a, 'unit circle'), (b, 'image under A')]:
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title(ttl)
    ax.set_xlabel('first component (dimensionless)'); ax.set_ylabel('second component (dimensionless)')
plt.tight_layout()
plt.savefig('least-squares_svd.pdf', bbox_inches='tight')
