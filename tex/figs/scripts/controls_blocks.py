import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
n, ell = 12, 3
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
for ax, circ in zip(axs, [False, True]):
    for i in range(n):
        ax.add_patch(Rectangle((i, 0), 0.9, 0.9, color='0.6'))
        ax.text(i + 0.45, 0.45, str(i + 1), ha='center', va='center', fontsize=7)
    nblocks = n if circ else n - ell + 1
    for j in range(nblocks):
        yy = -1.2 - 0.75*j
        for a in range(ell):
            pos = j + a
            col = 'tab:orange' if pos >= n else 'tab:blue'
            ax.add_patch(Rectangle((pos % n, yy), 0.9, 0.6, color=col))
            ax.text(pos % n + 0.45, yy + 0.3, str(pos % n + 1), ha='center', va='center', fontsize=6, color='w')
    ax.set_xlim(-0.2, n + 0.1)
    ax.set_ylim(-1.2 - 0.75*n, 1.1)
    ax.set_title('circular: %d blocks' % n if circ else 'moving: %d blocks' % (n - ell + 1), fontsize=9)
    ax.axis('off')
plt.savefig('controls_blocks.pdf', bbox_inches='tight')
