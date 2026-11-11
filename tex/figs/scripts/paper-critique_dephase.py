import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = np.asarray(qmnt.zeros(2000), dtype=float)
T = t[399] - t[0]          # window of zeros 1--400
dw = np.pi / T             # matching tolerance (upper bound)
ts = t[399]                # phases matched at t_400
w = np.log(2.0)

def wave(tt, shift):
    out = np.zeros_like(tt)
    for m in (1, 2, 3):
        out += np.sin(m * w * tt + m * shift * (tt - ts)) / (m * 2.0 ** (m / 2))
    return -out / np.pi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
tt = np.linspace(t[1000] - 20, t[1000] + 20, 1500)
ax1.plot(tt, wave(tt, 0.0), "k-", lw=1.2, label=r"exact, $\omega=\log 2$")
ax1.plot(tt, wave(tt, dw), "r--", lw=1.2, label=r"learned, $\omega+\pi/T$")
ax1.axvline(t[1000], color="gray", lw=0.8)
ax1.set_xlabel(r"height $t$ (dimensionless)")
ax1.set_ylabel("prime-2 wave (counts)")
ax1.set_title(r"near $t_{1001}$", fontsize=9)
ax1.legend(fontsize=7, loc="upper right")

tg = np.linspace(t[0], t[1999], 2000)
for m, st in zip((1, 2, 3), ("-", "--", ":")):
    ax2.plot(tg, m * dw * np.abs(tg - ts), st, color="C0", label="m = %d" % m)
for y in (np.pi / 2, np.pi):
    ax2.axhline(y, color="gray", lw=0.6, ls="-.")
for k in (399, 1000, 1999):
    ax2.axvline(t[k], color="gray", lw=0.6)
ax2.set_ylim(0, 15)
ax2.set_xlabel(r"height $t$ (dimensionless)")
ax2.set_ylabel("phase error (rad)")
ax2.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig("paper-critique_dephase.pdf", bbox_inches="tight")
