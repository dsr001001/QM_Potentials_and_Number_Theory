import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

n = np.arange(401, 501)
s = qmnt.smooth_zeros(n)
y = qmnt.Nfl(s, [2], mmax=3)          # RE2 sampled at smooth zeros
p1 = 0.5 * y
rng = np.random.default_rng(2)
p2 = rng.normal(0.0, y.std(), size=y.size)

fig, axs = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
axs[0].plot(n, y, "k-", lw=1.5, label="truth")
axs[0].plot(n, p1, "b--", lw=1.5, label="half amplitude")
axs[0].set_title("corr +1, rms = null/2")
axs[1].plot(n, y, "k-", lw=1.5, label="truth")
axs[1].plot(n, p2, "r.-", lw=0.8, label="independent noise")
axs[1].set_title("corr near 0, rms > null")
for ax in axs:
    ax.set_xlabel("level index $n$")
    ax.legend(fontsize=8, loc="lower left")
axs[0].set_ylabel("fluctuation (mean spacings)")
plt.tight_layout()
plt.savefig("probability_corr_rms.pdf", bbox_inches="tight")
