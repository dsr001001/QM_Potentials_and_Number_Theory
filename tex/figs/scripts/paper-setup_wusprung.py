import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

V0 = 3.10073*np.pi
def x_of_V(V):
    a = np.sqrt(V - V0); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*2*np.log((b + a)/np.sqrt(V0)))/np.pi

V = np.linspace(V0, 1600, 3000)
x = x_of_V(V)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, V, "k-"); ax.plot(-x, V, "k-")
for L, Vm, c in [(x_of_V(1000.0), 1000, "b"), (x_of_V(1500.0), 1500, "r")]:
    ax.axvline(L, color=c, ls="--", lw=0.8); ax.axvline(-L, color=c, ls="--", lw=0.8)
    ax.axhline(Vm, color=c, ls=":", lw=0.8)
    ax.text(L + 0.5, 150, "$L$=%.3f" % L, color=c, fontsize=8, rotation=90)
ax.set_xlabel("$x$ (units $\\hbar=2m=1$)")
ax.set_ylabel("$V(x)$ (units $\\hbar=2m=1$)")
ax.set_xlim(-66, 66); ax.set_ylim(0, 1650)
plt.savefig("paper-setup_wusprung.pdf", bbox_inches="tight")
