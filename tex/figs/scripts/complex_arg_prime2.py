import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t = np.linspace(0, 40, 4001)
x = np.exp(-(0.5 + 1j * t) * np.log(2))
phase = -np.angle(1 - x) / np.pi
re2 = -sum(np.sin(m * t * np.log(2)) / (m * 2 ** (m / 2)) for m in (1, 2, 3)) / np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, phase, "C0-", lw=1.6, label=r"$-\frac{1}{\pi}\mathrm{Arg}(1-2^{-1/2-it})$")
ax.plot(t, re2, "C1--", lw=1.4, label=r"RE2: $-\frac{1}{\pi}\sum_{m=1}^{3}\frac{\sin(mt\log 2)}{m2^{m/2}}$")
P = 2 * np.pi / np.log(2)
ax.annotate("", xy=(P, 0.3), xytext=(0, 0.3), arrowprops=dict(arrowstyle="<->"))
ax.text(1.5, 0.32, r"period $2\pi/\log 2\approx 9.065$", fontsize=8)
ax.axhline(0, color="gray", lw=0.6)
ax.set_ylim(-0.35, 0.42)
ax.set_xlabel(r"height $t$ (dimensionless)")
ax.set_ylabel("value (dimensionless)")
ax.legend(fontsize=8, loc="lower right")
plt.savefig("complex_arg_prime2.pdf", bbox_inches="tight")
