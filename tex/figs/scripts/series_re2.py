import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 40, 2000)
def w(M):
    return -sum(np.sin(m*t*np.log(2))/(m*2**(m/2)) for m in range(1, M + 1))/np.pi
bound = 0.048698
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(t, w(1), lw=0.8, label='$m=1$')
a1.plot(t, w(3), lw=1.2, label=r'$m\leq 3$')
a1.plot(t, w(40), 'k--', lw=0.8, label=r'$m\leq 40$')
a1.set_xlabel('height $t$ (dimensionless)')
a1.set_ylabel('wave (mean-spacing units)')
a1.legend(fontsize=7)
a2.plot(t, w(40) - w(3), lw=0.8)
a2.axhline(bound, color='k', ls='--', lw=0.8)
a2.axhline(-bound, color='k', ls='--', lw=0.8)
a2.set_xlabel('height $t$ (dimensionless)')
a2.set_ylabel(r'$m\leq40$ minus $m\leq3$')
plt.tight_layout()
plt.savefig("series_re2.pdf", bbox_inches="tight")
