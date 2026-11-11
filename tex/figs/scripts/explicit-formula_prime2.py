import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 50, 4000); l2 = np.log(2)
fig, ax = plt.subplots(figsize=(6, 4))
tot = np.zeros_like(t)
for m in (1, 2, 3):
    y = -np.sin(m*t*l2)/(np.pi*m*2**(m/2)); tot += y
    ax.plot(t, y, lw=0.8, label=f"m = {m}")
ax.plot(t, tot, 'k', lw=2, label="sum (RE2)")
for v in [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]:
    ax.axvline(v, ymin=0, ymax=0.06, color='r', lw=1.5)
ax.set_xlabel("height t (dimensionless)")
ax.set_ylabel("prime-2 wave (counts)")
ax.legend(fontsize=8, loc="upper right")
plt.savefig("explicit-formula_prime2.pdf", bbox_inches="tight")
