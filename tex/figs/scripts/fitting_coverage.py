import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
L = 44.883; t = qmnt.zeros(500); n = np.arange(1, 501)
plt.figure(figsize=(6, 4))
plt.plot(n, 2*np.sqrt(t), "k-", label=r"$2\sqrt{t_n}$")
for M in [300, 400, 600, 900]:
    plt.axhline(M*np.pi/L, ls="--", lw=1, label="M=%d" % M, color=plt.cm.viridis(M/1000))
plt.xlabel("level index n"); plt.ylabel("wavenumber (1/length)"); plt.legend(fontsize=8)
plt.savefig("fitting_coverage.pdf", bbox_inches="tight")
