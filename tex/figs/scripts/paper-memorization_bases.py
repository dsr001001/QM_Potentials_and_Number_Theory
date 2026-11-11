import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lab = ['cos 400','Cheb 200','Cheb 400','Cheb 600','Cheb ridge','hat 200','hat 400','hat 600']
r = np.array([0.369,0.368,0.291,0.733,0.264,0.342,0.340,0.330])
rlo = np.array([0.32,0.32,0.25,0.64,0.24,0.30,0.29,0.29])
rhi = np.array([0.42,0.42,0.33,0.82,0.29,0.38,0.39,0.37])
c = np.array([-0.07,-0.05,0.18,0.01,0.03,-0.05,-0.19,0.09])
clo = np.array([-0.22,-0.25,-0.02,-0.20,-0.21,-0.24,-0.37,-0.13])
chi = np.array([0.07,0.14,0.37,0.23,0.28,0.14,-0.01,0.30])
y = np.arange(len(lab))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
a1.errorbar(r, y, xerr=[r - rlo, rhi - r], fmt='o', capsize=3)
a1.axvline(0.263, ls='--', color='k')
a1.set_yticks(y); a1.set_yticklabels(lab, fontsize=7)
a1.set_xlabel('OOS rms (mean spacings)')
a2.errorbar(c, y, xerr=[c - clo, chi - c], fmt='s', capsize=3, color='C1')
a2.axvline(0.0, ls='--', color='k')
a2.set_xlabel('OOS correlation (dimensionless)')
plt.tight_layout()
plt.savefig('paper-memorization_bases.pdf', bbox_inches='tight')
