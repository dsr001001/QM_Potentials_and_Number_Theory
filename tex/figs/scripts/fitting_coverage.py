import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
L = 44.883
t = np.asarray(qmnt.zeros(500))
n = np.arange(1, 501)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, 2*np.sqrt(t), 'k-', label="2 sqrt(t_n)")
for M, col in zip([300, 400, 600, 900], ['C0', 'C1', 'C2', 'C3']):
    ax.axhline(M*np.pi/L, color=col, ls='--', label="M = %d" % M)
ax.set_xlabel("level index n")
ax.set_ylabel("wavenumber (inverse length, units hbar=2m=1)")
ax.legend(fontsize=8)
plt.savefig("fitting_coverage.pdf", bbox_inches="tight")
