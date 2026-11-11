import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

Q = 200
fig, ax = plt.subplots(figsize=(6, 4))
first = {"red": True, "blue": True, "gray": True}
labels = {"red": "six-prime oracle ($p\\leq13$, $m\\leq3$)",
          "blue": "added in 25-prime oracle ($p\\leq97$, $m\\leq3$)",
          "gray": "$m\\geq4$ (not in Eq. (4))"}
cols = {"red": "C3", "blue": "C0", "gray": "0.6"}
for p in qmnt.primes_upto(Q):
    p = int(p); m = 1; q = p
    while q <= Q:
        w = 1.0 / (m * p ** (m / 2.0))
        if m >= 4:
            key = "gray"
        elif p <= 13:
            key = "red"
        else:
            key = "blue"
        ax.vlines(np.log(q), 0, w, color=cols[key], lw=1.4,
                  label=labels[key] if first[key] else None)
        first[key] = False
        m += 1; q *= p
ax.set_xlabel(r"angular frequency in $t$, $\omega=\log q$ (dimensionless)")
ax.set_ylabel(r"weight $\Lambda(q)/(\sqrt{q}\,\log q)=1/(m\,p^{m/2})$")
ax.set_xlim(0.5, np.log(Q) + 0.1); ax.set_ylim(0, 0.75)
for q, txt in [(2, "2"), (3, "3"), (4, "4"), (5, "5"), (16, "16")]:
    m = {2: 1, 3: 1, 4: 2, 5: 1, 16: 4}[q]; p = {2: 2, 3: 3, 4: 2, 5: 5, 16: 2}[q]
    ax.text(np.log(q), 1.0 / (m * p ** (m / 2.0)) + 0.015, txt, ha="center", fontsize=7)
ax.legend(fontsize=7, loc="upper right")
plt.savefig("primes_spectrum.pdf", bbox_inches="tight")
