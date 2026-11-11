import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
def box(x, y, txt):
    ax.text(x, y, txt, ha='center', va='center', fontsize=7, bbox=dict(boxstyle='round', fc='w', ec='k'))
def arr(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->'))
box(1.2, 3.0, 'zeros $t_n$\n(mpmath, step 1)')
box(3.6, 5.0, 'smooth zeros $s_n$, gap$_n$\n(Eq. 1, step 2)')
box(3.6, 1.0, 'Wu-Sprung $V$ (Eq. 2)\ngrid, solver, $\\Delta_n$ (steps 3-5)')
box(6.3, 1.0, 'fit $\\delta V$ on training zeros\n(steps 6-7)')
box(6.3, 5.0, 'oracle Eq. 4\nno fitted parameters (step 8)')
box(8.8, 3.0, 'score held-out zeros:\nrms, corr vs null')
arr(1.9, 3.4, 2.7, 4.7); arr(1.9, 2.6, 2.7, 1.3)
arr(4.9, 5.0, 5.3, 5.0); arr(5.0, 1.0, 5.2, 1.0)
arr(4.0, 4.6, 5.6, 1.4)
arr(7.4, 4.7, 8.3, 3.4); arr(7.4, 1.3, 8.3, 2.6)
plt.savefig('paper-setup_pipeline.pdf', bbox_inches='tight')
