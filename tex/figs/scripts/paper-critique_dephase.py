import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t1, t500, t1000 = 14.1347251417, 811.1843588465, 1419.42248094599   # paper anchors
T = t500 - t1
dw = np.pi / T
w = np.log(2.0)
t = np.linspace(600, t1000, 3000)
def wave(t, om, tstar):
    s = 0.0
    for m in (1, 2, 3):
        # phase chosen so that the learned wave agrees with the true one at tstar
        s = s + np.sin(m * (om * (t - tstar) + w * tstar)) / (m * 2 ** (m / 2))
    return -s / np.pi
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(t, wave(t, w, t500), "k-", lw=1, label="exact log 2")
a1.plot(t, wave(t, w + dw, t500), "r--", lw=1, label="log 2 + pi/T")
a1.axvline(t500, color="0.5")
a1.set_xlim(1100, t1000)
a1.set_xlabel("height t"); a1.set_ylabel("fluctuation (counts)")
a1.legend(fontsize=7, loc="lower left")
for m, c in zip((1, 2, 3), ("k", "b", "r")):
    a2.plot(t[t >= t500], m * dw * (t[t >= t500] - t500), c, label=f"m={m}")
a2.axhline(np.pi / 2, ls=":", color="0.4"); a2.axhline(np.pi, ls="--", color="0.4")
a2.set_xlabel("height t"); a2.set_ylabel("phase error (rad)")
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("paper-critique_dephase.pdf", bbox_inches="tight")
