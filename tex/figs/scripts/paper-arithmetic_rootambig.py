import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

s = 680.0
g = 2*np.pi/np.log(s/(2*np.pi))
t = np.linspace(s - 3*g, s + 3*g, 2000)
fig, ax = plt.subplots(figsize=(6, 4))
for A, c in ((0.05, "tab:blue"), (0.3, "tab:red")):
    G = qmnt.Nsm(t) - qmnt.Nsm(s) + A*np.sin(5*t)
    ax.plot(t, G, color=c, label=f"A = {A}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("height t")
ax.set_ylabel("G(t) (counts)")
ax.legend()
plt.savefig("paper-arithmetic_rootambig.pdf", bbox_inches="tight")
