import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def sz(n):
    try:
        out = np.asarray(qmnt.smooth_zeros(n), dtype=float)
        if out.shape != n.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.smooth_zeros(int(k))) for k in n])

def wsx(V):
    try:
        out = np.asarray(qmnt.wu_sprung_x(V), dtype=float)
        if out.shape != V.shape:
            raise ValueError
        return out
    except Exception:
        return np.array([float(qmnt.wu_sprung_x(v)) for v in V])

n = np.arange(1, 1001)
s = sz(n)
x = wsx(s)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, x, "k-", lw=1.2, label="turning point x(s_n)")
ax.axhline(44.883, color="tab:blue", ls="--", lw=1, label="L = 44.883 (scale 1)")
ax.axhline(59.957, color="tab:green", ls="--", lw=1, label="L = 59.957 (scale 2)")
ax.axvspan(201, 300, color="tab:orange", alpha=0.25, label="gap 201-300")
ax.axvspan(401, 500, color="tab:purple", alpha=0.2, label="holdout 401-500")
ax.set_xlabel("level index n")
ax.set_ylabel("turning point x (units hbar = 2m = 1)")
ax.legend(fontsize=7, loc="lower right")
plt.savefig("paper-critique_turning.pdf", bbox_inches="tight")
