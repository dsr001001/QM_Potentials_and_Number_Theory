import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V0 = 3.10073*np.pi
def xws(V):
    a = np.sqrt(V-V0); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b+a)/(b-a)))/np.pi
aa = np.linspace(1e-6, np.sqrt(1700-V0), 20000)
VV = V0 + aa**2
XX = xws(VV)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(XX, VV, "k", label="Wu--Sprung, Eq. (2)")
xs = np.linspace(0, 6, 100)
c = (np.pi/np.log(V0/(2*np.pi)))**2
ax.plot(xs, V0 + c*xs**2, "g:", label="$V_0+51.3x^2$")
Va = np.linspace(30, 1700, 400)
ax.plot(np.sqrt(Va)/np.pi*(np.log(2*Va/np.pi)-2), Va, "r--", label="large-$V$ form")
for L in [44.883, 59.957]:
    ax.axvline(L, color="gray", lw=0.8)
ax.set_xlim(0, 62); ax.set_ylim(0, 1700)
ax.set_xlabel("$x$ (units $\\hbar=2m=1$)"); ax.set_ylabel("$V(x)$")
ax.legend(fontsize=8)
plt.savefig("wu-sprung_potential.pdf", bbox_inches="tight")
