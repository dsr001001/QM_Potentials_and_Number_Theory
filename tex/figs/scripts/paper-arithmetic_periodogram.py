import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

t = np.asarray(qmnt.zeros(400)); n = np.arange(1, 401)
y = n - 0.5 - qmnt.Nsm(t)
w = np.linspace(0.05, 4, 4000)
P = np.abs(np.exp(-1j*np.outer(w, t)) @ y)**2
P /= P.max()
fig, ax = plt.subplots(figsize=(6, 4))
pp = [q for q in range(2, 51)
      if len({p for p in qmnt.primes_upto(q) if q % p == 0}) == 1]
for q in pp:
    ax.axvline(np.log(q), color="0.8", lw=0.7)
    if q <= 16:
        ax.text(np.log(q), 1.02, str(q), fontsize=6, ha="center")
ax.plot(w, P, color="k", lw=0.8)
ax.set_xlabel("angular frequency omega (1/t units)")
ax.set_ylabel("normalized periodogram")
ax.set_ylim(0, 1.1)
plt.savefig("paper-arithmetic_periodogram.pdf", bbox_inches="tight")
