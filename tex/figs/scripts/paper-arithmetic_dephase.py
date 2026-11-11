import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

a = 1/(np.pi*np.sqrt(2)); w = np.log(2); dw = 2e-3; tc = 347.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for ax, (lo, hi) in ((a1, (320, 375)), (a2, (1975, 2030))):
    t = np.linspace(lo, hi, 800)
    ax.plot(t, a*np.sin(w*t), "k", label="true")
    ax.plot(t, a*np.sin(w*t + dw*(t - tc)), "--", color="tab:orange", label="learned")
    ax.set_xlabel("height t")
a1.set_ylabel("term value (mean spacings)")
a1.legend(fontsize=7)
plt.tight_layout()
plt.savefig("paper-arithmetic_dephase.pdf", bbox_inches="tight")
