import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
boxes = {'I': (0.5, 0.9, 'I  Mathematical\nfoundations'), 'II': (0.2, 0.65, 'II  Quantum\nmechanics'), 'III': (0.8, 0.65, 'III  Primes and\nzeta'), 'IV': (0.5, 0.42, 'IV  Spectra, randomness,\noperators'), 'V': (0.5, 0.22, 'V  Computation, fitting,\ngeneralization'), 'VI': (0.5, 0.04, 'VI  The paper')}
for k, (xb, yb, s) in boxes.items():
    ax.text(xb, yb, s, ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='k'))
def arr(a, b):
    xa, ya, _ = boxes[a]; xb, yb, _ = boxes[b]
    ax.annotate('', xy=(xb, yb + 0.06), xytext=(xa, ya - 0.06), arrowprops=dict(arrowstyle='->'))
for a, b in [('I', 'II'), ('I', 'III'), ('II', 'IV'), ('III', 'IV'), ('IV', 'V'), ('V', 'VI')]:
    arr(a, b)
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0); ax.axis('off')
plt.savefig('intro_roadmap.pdf', bbox_inches='tight')
