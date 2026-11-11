import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
z = 3 + 2j
pts = [(z, r"$z=3+2i$", "C0", (0.15, 0.15)),
       (np.conj(z), r"$\bar z=3-2i$", "C1", (0.15, -0.35)),
       (-z, r"$-z=-3-2i$", "C2", (-0.3, -0.45)),
       (1j * z, r"$iz=-2+3i$", "C3", (0.15, 0.15))]
for w, lab, c, off in pts:
    ax.annotate("", xy=(w.real, w.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.plot(w.real, w.imag, "o", color=c)
    ax.text(w.real + off[0], w.imag + off[1], lab, color=c)
th = np.linspace(0, np.angle(z), 50)
ax.plot(0.9 * np.cos(th), 0.9 * np.sin(th), "k-", lw=1)
ax.text(1.0, 0.2, r"$\theta=\arg z$")
ax.plot([3, 3], [0, 2], "k:", lw=1)
ax.plot([0, 3], [2, 2], "k:", lw=1)
ax.text(1.0, 1.05, r"$|z|=\sqrt{13}$", rotation=np.degrees(np.angle(z)))
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlim(-4.5, 5)
ax.set_ylim(-3, 4)
ax.set_aspect("equal")
ax.set_xlabel(r"real part $\mathrm{Re}\,z$ (dimensionless)")
ax.set_ylabel(r"imaginary part $\mathrm{Im}\,z$ (dimensionless)")
plt.savefig("complex_plane.pdf", bbox_inches="tight")
