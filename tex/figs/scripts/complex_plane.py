import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
pts = [(r"$z=3+2i$", 3 + 2j, "C0"), (r"$\bar z=3-2i$", 3 - 2j, "C1"),
       (r"$-z$", -3 - 2j, "C2"), (r"$iz=-2+3i$", -2 + 3j, "C3")]
for lab, w, c in pts:
    ax.annotate("", xy=(w.real, w.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.plot(w.real, w.imag, "o", color=c)
    ax.text(w.real + 0.15, w.imag + 0.15, lab, color=c)
th = np.arctan2(2, 3)
a = np.linspace(0, th, 50)
ax.plot(0.9 * np.cos(a), 0.9 * np.sin(a), "k-")
ax.text(1.0, 0.2, r"$\theta$")
ax.text(1.0, 1.15, r"$|z|=\sqrt{13}$", rotation=np.degrees(th))
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-4, 4.8); ax.set_ylim(-3, 4); ax.set_aspect("equal")
ax.set_xlabel("real part (dimensionless)"); ax.set_ylabel("imaginary part (dimensionless)")
ax.grid(alpha=0.3)
plt.savefig("complex_plane.pdf", bbox_inches="tight")
