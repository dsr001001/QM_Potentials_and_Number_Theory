import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(1, 8.5, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for k in range(1, 8):
    a1.bar(k, 1.0/k, width=1, align='edge', color='C0', alpha=0.35, edgecolor='C0')
for k in range(2, 9):
    a2.bar(k - 1, 1.0/k, width=1, align='edge', color='C1', alpha=0.35, edgecolor='C1')
for ax in (a1, a2):
    ax.plot(x, 1/x, 'k', lw=1.5, label=r'$f(x)=1/x$')
    ax.set_xlabel('$x$ (dimensionless)')
    ax.set_xlim(0.8, 8.7)
    ax.legend(fontsize=8)
a1.set_ylabel('$f(x)$ (dimensionless)')
a1.set_title(r'height $f(k)$ on $[k,k+1]$: above', fontsize=9)
a2.set_title(r'height $f(k)$ on $[k-1,k]$: below', fontsize=9)
plt.savefig("series_integraltest.pdf", bbox_inches="tight")
