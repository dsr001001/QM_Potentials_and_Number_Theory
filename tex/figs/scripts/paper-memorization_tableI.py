import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Paper Table I, quoted exactly: (scale, in-sample, OOS, corr)
rows = [(1,0.212,0.317,-0.01),(1,0.120,0.320,-0.17),(1,0.128,0.273,0.04),
        (1,0.226,0.274,-0.09),(1,0.137,0.369,-0.07),(1,0.142,0.271,-0.16),
        (1,0.206,0.265,-0.14),(1,0.064,0.313,-0.16),(1,0.147,0.439,0.12),
        (2,0.242,0.370,-0.01),(2,0.170,0.279,-0.03),(2,0.103,0.302,0.08)]
r = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].axhspan(0.261, 0.274, color="0.85", label="null range")
m1 = r[:,0]==1; m2 = r[:,0]==2
ax[0].plot(r[m1,1], r[m1,2], "o", color="C0", label="scale 1")
ax[0].plot(r[m2,1], r[m2,2], "s", color="C3", label="scale 2")
ax[0].set_xlabel("in-sample rms (mean spacings)")
ax[0].set_ylabel("OOS rms (mean spacings)")
ax[0].set_xlim(0, 0.3); ax[0].set_ylim(0, 0.5)
ax[0].legend(fontsize=7, loc="lower right")
idx = np.arange(1, 13)
ax[1].bar(idx, r[:,3], color=["C0" if s==1 else "C3" for s in r[:,0]])
ax[1].axhline(0.96, ls="--", color="k", lw=1, label="6-prime oracle (+0.96)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("row of Table I")
ax[1].set_ylabel("OOS correlation")
ax[1].set_ylim(-0.3, 1.05)
ax[1].set_xticks(idx)
ax[1].legend(fontsize=7, loc="center right")
plt.tight_layout()
plt.savefig("paper-memorization_tableI.pdf", bbox_inches="tight")
