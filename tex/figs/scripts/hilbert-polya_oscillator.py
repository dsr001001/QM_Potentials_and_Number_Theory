import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E = np.linspace(0.0, 5.0, 4001)
N = np.floor(E + 0.5)
def trace(E, R):
    s = np.zeros_like(E)
    for r in range(1, R + 1):
        s += (-1)**r * np.sin(2*np.pi*r*E) / r
    return E + s/np.pi
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(E, N, color="k", lw=1.5, label="staircase $N(E)$")
for R, c in [(1, "C0"), (4, "C1"), (30, "C3")]:
    ax.plot(E, trace(E, R), color=c, lw=1, label=f"trace sum, $R={R}$")
ax.plot(E, E, "k--", lw=0.8, label="smooth part $E$")
ax.set_xlabel(r"energy $E$ (units of $\hbar\omega$)")
ax.set_ylabel("number of levels $N(E)$")
ax.legend(fontsize=8)
plt.savefig("hilbert-polya_oscillator.pdf", bbox_inches="tight")
