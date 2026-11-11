import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a1 = np.array([1, 0, 1.]); a2 = np.array([0, 1, 1.]); y = np.array([1, 2, 0.]); yh = a2.copy()
fig = plt.figure(figsize=(6, 4)); ax = fig.add_subplot(111, projection='3d')
s, u = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 2.0, 10))
ax.plot_surface(s*a1[0] + u*a2[0], s*a1[1] + u*a2[1], s*a1[2] + u*a2[2], alpha=0.25, color='gray')
def arrow(v, col, lab):
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=col, arrow_length_ratio=0.1)
    ax.text(1.08*v[0], 1.08*v[1], 1.08*v[2], lab, color=col)
arrow(a1, 'k', 'a1'); arrow(a2, 'k', 'a2'); arrow(y, 'b', 'y'); arrow(yh, 'g', 'y-hat')
ax.plot([yh[0], y[0]], [yh[1], y[1]], [yh[2], y[2]], 'r--')
m = (y + yh)/2; ax.text(m[0], m[1], m[2], 'residual', color='r')
ax.set_xlabel('component 1'); ax.set_ylabel('component 2'); ax.set_zlabel('component 3')
plt.savefig('least-squares_projection.pdf', bbox_inches='tight')
