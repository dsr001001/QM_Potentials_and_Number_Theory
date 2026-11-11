import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(1000); n = np.arange(1, 1001)
y = n - 0.5 - qmnt.Nsm(t); y = y - y.mean()
w = np.linspace(0.3, 3.6, 6000)
P = np.array([abs(np.sum(y*np.exp(-1j*v*t)))**2/len(t) for v in w])
plt.figure(figsize=(6, 4)); plt.plot(w, P, lw=0.8)
qs = [q for q in range(2, 37) if len({p for p in qmnt.primes_upto(q) if q % p == 0}) == 1]
for q in qs:
    plt.axvline(np.log(q), color='gray', ls='--', lw=0.5)
for q in (2, 3, 4, 5, 7):
    plt.text(np.log(q), P.max()*1.02, str(q), ha='center', fontsize=8)
plt.xlabel(r"$\omega$ (rad per unit $t$)"); plt.ylabel(r"$P(\omega)$")
plt.savefig("fourier-transform_zerospec.pdf", bbox_inches="tight")
