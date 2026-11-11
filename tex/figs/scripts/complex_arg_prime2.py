import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 40, 4001)
x = 2.0 ** (-0.5 - 1j * t)
phase = -np.angle(1 - x) / np.pi
re2 = -sum(np.sin(m * t * np.log(2)) / (m * 2 ** (m / 2)) for m in (1, 2, 3)) / np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, phase, "C0-", label=r"$-\frac{1}{\pi}\mathrm{Arg}(1-2^{-1/2-it})$")
ax.plot(t, re2, "C3--", label="RE2 (three terms)")
P2 = 2 * np.pi / np.log(2)
for k in range(1, 5):
    ax.axvline(k * P2, color="gray", lw=0.6, ls=":")
ax.text(P2 + 0.3, 0.33, r"$2\pi/\log 2\approx 9.065$", fontsize=8)
ax.set_xlabel(r"height $t$ (dimensionless)"); ax.set_ylabel("value (dimensionless)")
ax.legend(fontsize=8, loc="lower right")
plt.savefig("complex_arg_prime2.pdf", bbox_inches="tight")
