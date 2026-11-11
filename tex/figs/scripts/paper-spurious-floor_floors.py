import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(6.5,3), sharey=True)
lab = ["PT-odd", "abs.-even"]
ax[0].plot([0,1],[0.2321,0.2358],"o",label="surrogate floor")
ax[0].plot([1],[0.2322],"o",mfc="none",label="even, before repair")
ax[0].plot([0,1],[0.2523,0.2477],"s",label="exact spectrum")
ax[0].set_title("unconstrained")
ax[1].plot([0,1],[0.2335,0.2422],"o",label="surrogate, R=4")
ax[1].plot([0,1],[0.2391,0.2423],"s",label="R=4 exact")
ax[1].set_title("amplitude controlled")
for a in ax:
    a.axhline(0.2438, ls="--", c="k"); a.set_xticks([0,1]); a.set_xticklabels(lab)
    a.set_xlim(-0.5,1.5); a.legend(fontsize=7)
ax[0].set_ylabel("in-sample rms (mean spacings)")
plt.savefig("paper-spurious-floor_floors.pdf", bbox_inches="tight")
