import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
u = qmnt.Nsm(qmnt.zeros(10000))
d = []
for k in range(1, 8):
    dk = u[k:] - u[:-k]; d.append(dk[dk < 3])
d = np.concatenate(d)
bw = 0.05; bins = np.arange(0, 3 + bw, bw)
c, e = np.histogram(d, bins=bins)
v = 0.5*(e[1:] + e[:-1])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(v, c/(len(u)*bw), width=bw, alpha=0.5, label="zeros")
vv = np.linspace(1e-3, 3, 400)
ax.plot(vv, 1 - (np.sin(np.pi*vv)/(np.pi*vv))**2, label="Montgomery / GUE")
ax.axhline(1, ls="--", color="k", label="Poisson")
ax.set_xlabel("separation v (mean spacings)"); ax.set_ylabel("R2(v)"); ax.legend()
plt.savefig("level-statistics_paircorr.pdf", bbox_inches="tight")
