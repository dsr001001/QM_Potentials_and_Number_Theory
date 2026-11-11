import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
pos = {'I': (0.5, 0.92), 'II': (0.22, 0.70), 'III': (0.78, 0.70), 'IV': (0.5, 0.48), 'V': (0.5, 0.27), 'VI': (0.5, 0.06)}
lab = {'I': 'Part I: mathematical foundations', 'II': 'Part II: quantum mechanics', 'III': 'Part III: primes and zeta', 'IV': 'Part IV: spectra, randomness, operators', 'V': 'Part V: computation, fitting, generalization', 'VI': 'Part VI: the paper (uses Parts I-V)'}
for k, (xp, yp) in pos.items():
    ax.text(xp, yp, lab[k], ha='center', va='center', fontsize=8, bbox=dict(boxstyle='round', fc='w'))
edges = [('I', 'II'), ('I', 'III'), ('II', 'IV'), ('III', 'IV'), ('II', 'V'), ('IV', 'V'), ('V', 'VI')]
for a, b in edges:
    ax.annotate('', xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle='->', shrinkA=12, shrinkB=12))
plt.savefig('intro_roadmap.pdf', bbox_inches='tight')
