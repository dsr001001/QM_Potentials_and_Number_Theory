import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
V0 = 3.10073*np.pi
def Nsm(t): u = t/(2*np.pi); return u*np.log(u) - u + 7/8
def xws(V):
    a = np.sqrt(V-V0); b = np.sqrt(V)
    return (a*np.log(V0/(2*np.pi*np.e**2)) + b*np.log((b+a)/(b-a)))/np.pi
n = np.arange(1, 1001)
s = np.array([brentq(lambda t: Nsm(t)-(k-0.5), 7.0, 3000.0) for k in n])
xn = xws(s)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n, xn, "k")
ax.axvspan(201, 300, color="orange", alpha=0.4, label="levels 201--300")
ax.axvspan(401, 500, color="skyblue", alpha=0.5, label="levels 401--500")
for L in [44.883, 59.957]:
    ax.axhline(L, color="gray", ls="--", lw=0.8)
ax.set_xlabel("level index $n$"); ax.set_ylabel("turning point $x_n$ (units $\\hbar=2m=1$)")
ax.legend(fontsize=8)
plt.savefig("wu-sprung_turning.pdf", bbox_inches="tight")
