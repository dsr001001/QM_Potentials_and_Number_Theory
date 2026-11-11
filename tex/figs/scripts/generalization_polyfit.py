import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
t_tr = np.linspace(0, 2*np.pi, 10)
y_tr = np.sin(t_tr) + 0.15*rng.standard_normal(10)
tt = np.linspace(0, 2.6*np.pi, 800)

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(2*np.pi, 2.6*np.pi, color="0.9", label="extrapolation region")
ax.plot(tt, np.sin(tt), "k--", lw=1, label=r"truth $\sin t$")
for deg, col in [(1, "C0"), (3, "C1"), (9, "C3")]:
    p = np.polynomial.Polynomial.fit(t_tr, y_tr, deg)
    ax.plot(tt, p(tt), color=col, lw=1.5, label=f"degree {deg}")
ax.plot(t_tr, y_tr, "ko", ms=5, label="training points")
ax.set_ylim(-3, 3)
ax.set_xlim(0, 2.6*np.pi)
ax.set_xlabel(r"$t$ (dimensionless)")
ax.set_ylabel(r"$y$ (dimensionless)")
ax.legend(loc="lower left", fontsize=8)
plt.savefig("generalization_polyfit.pdf", bbox_inches="tight")
