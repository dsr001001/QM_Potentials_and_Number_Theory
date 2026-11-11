import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
k = np.arange(1, 51)
a = 0.05
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, 1 - (1 - a)**k, "o-", ms=3, label="each test at alpha = 0.05")
ax.plot(k, 1 - (1 - a/k)**k, "s-", ms=3, label="Bonferroni, alpha/k")
ax.axhline(0.05, color="k", ls=":")
ax.set_xlabel("number of independent tests k")
ax.set_ylabel("P(at least one false positive)")
ax.legend(fontsize=8)
plt.savefig("controls_posthoc.pdf", bbox_inches="tight")
