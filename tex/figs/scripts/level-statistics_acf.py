import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401)
d = qmnt.fluct(qmnt.zeros(400), n); d = d - d.mean()
r = [np.mean(d[k:]*d[:-k])/np.mean(d*d) for k in range(1, 16)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 16), r); ax.axhline(0, color="k")
ax.set_xlabel("lag k (levels)"); ax.set_ylabel("autocorrelation r_k")
plt.savefig("level-statistics_acf.pdf", bbox_inches="tight")
