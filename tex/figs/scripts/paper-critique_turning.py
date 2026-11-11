import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
def sz(n):
    n = np.atleast_1d(np.asarray(n))
    try:
        out = np.asarray(qmnt.smooth_zeros(n), dtype=float)
        if out.shape != n.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.smooth_zeros(int(k))) for k in n])
def wsx(V):
    V = np.atleast_1d(np.asarray(V, dtype=float))
    try:
        out = np.asarray(qmnt.wu_sprung_x(V), dtype=float)
        if out.shape != V.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.wu_sprung_x(v)) for v in V])
n = np.arange(1, 1001)
x = wsx(sz(n))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, x, "k-")
ax.axvspan(201, 300, color="orange", alpha=0.3, label="gap 201-300")
ax.axvspan(401, 500, color="blue", alpha=0.2, label="holdout 401-500")
ax.axhline(qmnt.L_of(1000), ls="--", color="r", label="L = 44.883 (scale 1)")
ax.axhline(qmnt.L_of(1500), ls=":", color="m", label="L = 59.957 (scale 2)")
ax.set_xlabel("level index n")
ax.set_ylabel("turning point x(s_n) (units hbar = 2m = 1)")
ax.legend(fontsize=8, loc="lower right")
plt.savefig("paper-critique_turning.pdf", bbox_inches="tight")
