import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
y = np.linspace(0.02, 2.5, 600)
g = y*np.log(y) - y + 7/8
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(y, g, "k")
ax.axhline(0, color="gray", lw=0.8)
r1 = brentq(lambda u: u*np.log(u)-u+7/8, 0.05, 1.0)
r2 = brentq(lambda u: u*np.log(u)-u+7/8, 1.0, 3.0)
ax.plot([r1], [0], "rx", ms=9, label="rejected root ($V_0<2\\pi$)")
ax.plot([r2], [0], "go", ms=7, label="consistent root $y\\approx%.4f$" % r2)
yp = 3.10073/2
ax.plot([yp], [yp*np.log(yp)-yp+7/8], "b^", ms=7, label="paper $V_0=3.10073\\pi$")
ax.set_xlabel("$y=V_0/2\\pi$ (dimensionless)"); ax.set_ylabel("$N_{\\rm sm}(2\\pi y)$")
ax.legend(fontsize=8)
plt.savefig("wu-sprung_v0func.pdf", bbox_inches="tight")
