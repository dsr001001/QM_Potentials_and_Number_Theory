import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

t = np.linspace(0, 60, 3000)
l2 = np.log(2.0)
w = -(1/np.pi) * sum(np.sin(m*t*l2) / (m * 2**(m/2)) for m in (1, 2, 3))
P = 2*np.pi/l2

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, w, color="tab:red", lw=1.5)
ax.axhline(0, color="k", lw=0.6)
for k in range(1, 3):
    ax.axvline(k*P, color="gray", ls="--", lw=0.8)
ax.annotate("", xy=(2*P, 0.33), xytext=(P, 0.33),
            arrowprops=dict(arrowstyle="<->"))
ax.text(1.5*P, 0.35, r"period $2\pi/\log 2 \approx 9.065$", ha="center")
ax.set_xlabel(r"height $t$ (dimensionless)")
ax.set_ylabel("prime-2 wave (counting units)")
ax.set_xlim(0, 60)
ax.set_ylim(-0.4, 0.42)
ax.grid(alpha=0.3)
plt.savefig("intro_prime2.pdf", bbox_inches="tight")
