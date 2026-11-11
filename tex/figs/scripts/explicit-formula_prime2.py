import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = np.linspace(0, 40, 4000)
L2 = np.log(2.0)
fig, ax = plt.subplots(figsize=(6, 4))
total = np.zeros_like(t)
for m, st in [(1, "--"), (2, "-."), (3, ":")]:
    term = -np.sin(m * t * L2) / (np.pi * m * 2 ** (m / 2))
    total += term
    ax.plot(t, term, st, lw=1, label=f"m = {m}")
ax.plot(t, total, "k-", lw=2, label="sum (RE2)")
z = qmnt.zeros(5)
for tz in z:
    ax.axvline(tz, color="gray", lw=0.6, ymax=0.08)
P0 = 2 * np.pi / L2
ax.annotate("", xy=(P0 + 2, -0.37), xytext=(2, -0.37),
            arrowprops=dict(arrowstyle="<->"))
ax.text(2 + P0 / 2, -0.35, "period 9.065", ha="center", fontsize=8)
ax.set_xlabel("height t (dimensionless)")
ax.set_ylabel("contribution to count (mean spacings)")
ax.set_ylim(-0.42, 0.42)
ax.legend(fontsize=8, loc="upper right")
plt.savefig("explicit-formula_prime2.pdf", bbox_inches="tight")
