import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(1, 401)
d = qmnt.fluct(qmnt.zeros(400), n)
ws = [3, 5, 7, 9, 13]
bl = [np.sqrt(np.mean(np.convolve(d, np.ones(w)/w, mode="valid")**2)) for w in ws]
plt.figure(figsize=(6,4))
plt.plot(ws, bl, "o-", label="qmnt, $(t_n-s_n)/gap_n$")
plt.plot(ws, [0.108,0.049,0.033,0.035,0.019], "x", ms=9, label="paper Sec. V A")
plt.axhline(0.232, ls="--", c="k", label="floor 0.232")
plt.axhline(np.sqrt(np.mean(d**2)), ls=":", c="gray", label="rms of $d_n$")
plt.xlabel("window w (levels)"); plt.ylabel("block-coherent rms (mean spacings)")
plt.legend(fontsize=8)
plt.savefig("paper-spurious-floor_blocks.pdf", bbox_inches="tight")
