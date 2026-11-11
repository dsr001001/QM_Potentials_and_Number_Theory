import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
rng = np.random.default_rng(1)
W = 30
pois = np.sort(rng.uniform(0, W, W))
u = qmnt.Nsm(qmnt.zeros(1040)[1000:])
u = u - u[0] + 0.5
u = u[u < W]
fence = np.arange(0.5, W, 1.0)
fig, ax = plt.subplots(figsize=(6, 4))
for y, pts, lab in [(2, pois, "Poisson"), (1, u, "zeros (unfolded)"), (0, fence, "picket fence")]:
    ax.vlines(pts, y - 0.35, y + 0.35)
    ax.text(-0.5, y, lab, ha="right", va="center")
ax.set_xlim(-9, W); ax.set_yticks([])
ax.set_xlabel("unfolded position (units of mean spacing)")
plt.savefig("level-statistics_barcodes.pdf", bbox_inches="tight")
