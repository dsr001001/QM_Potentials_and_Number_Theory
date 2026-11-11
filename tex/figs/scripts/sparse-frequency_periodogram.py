import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(400); n = np.arange(1, 401)
u = t/(2*np.pi); y = n - 0.5 - (u*np.log(u) - u + 7/8)
w = np.linspace(0.05, 4.5, 20000)
C = np.cos(np.outer(w, t)) @ y; S = np.sin(np.outer(w, t)) @ y
Pw = (C**2 + S**2)/len(t)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(w, Pw, lw=0.6, color="k")
top = Pw.max()*1.05
for p in (2, 3, 5, 7, 11, 13):
    ax.axvline(np.log(p), ls="--", lw=0.7, color="tab:blue")
    ax.text(np.log(p), top, str(p), ha="center", fontsize=8, color="tab:blue")
for q in (4, 8, 9):
    ax.axvline(np.log(q), ls=":", lw=0.8, color="tab:red")
    ax.text(np.log(q), top*0.93, str(q), ha="center", fontsize=8, color="tab:red")
ax.set_ylim(0, top*1.1)
ax.set_xlabel(r"angular frequency $\omega$ (radians per unit height $t$)")
ax.set_ylabel(r"periodogram $\mathcal{P}(\omega)$ (counting units$^2$)")
plt.savefig("sparse-frequency_periodogram.pdf", bbox_inches="tight")
