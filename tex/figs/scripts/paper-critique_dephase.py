import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
z = qmnt.zeros(1000)
t1, t500, t1000 = z[0], z[499], z[999]
T = t500 - t1
dw = np.pi / T
w = np.log(2)

def wave(t, om, ts):
    out = np.zeros_like(t)
    for m in (1, 2, 3):
        phase = m * om * t - m * (om - w) * ts   # equals m*w*ts at t = ts
        out += np.sin(phase) / (m * 2 ** (m / 2))
    return -out / np.pi

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
tt = np.linspace(760, 1420, 4000)
a1.plot(tt, wave(tt, w, t500), "k-", lw=1, label="exact")
a1.plot(tt, wave(tt, w + dw, t500), "r--", lw=1, label="learned")
a1.axvline(t500, color="0.5", lw=0.8)
a1.set_xlim(1300, 1420)
a1.set_xlabel("height t")
a1.set_ylabel("prime-2 wave (zeros)")
a1.legend(fontsize=7, loc="upper left")
td = np.linspace(t500, t1000, 400)
for m in (1, 2, 3):
    a2.plot(td, m * dw * (td - t500), label="m=%d" % m)
a2.axhline(np.pi / 2, color="0.5", ls=":", lw=0.8)
a2.axhline(np.pi, color="0.5", ls="--", lw=0.8)
a2.set_xlabel("height t")
a2.set_ylabel("phase error (rad)")
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("paper-critique_dephase.pdf", bbox_inches="tight")
