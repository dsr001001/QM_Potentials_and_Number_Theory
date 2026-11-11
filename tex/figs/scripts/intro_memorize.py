import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4], float)
y = np.array([1.0, 2.9, 5.2, 6.9, 9.1])
p4 = np.polyfit(x, y, 4)
xs = np.linspace(-0.3, 5.3, 400)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xs, 1.0 + 2.025*xs, "k--", lw=1.3, label="line through first and last points")
ax.plot(xs, np.polyval(p4, xs), color="tab:red", lw=1.5, label="degree-4 polynomial through all points")
ax.plot(x, y, "o", color="tab:blue", ms=7, label="training data")
ax.plot([5], [11.0], "s", color="tab:green", ms=8, label="later measurement")
ax.axvspan(4.5, 5.3, color="gray", alpha=0.15)
ax.set_xlabel("input $x$ (arbitrary units)")
ax.set_ylabel("output $y$ (arbitrary units)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("intro_memorize.pdf", bbox_inches="tight")
