import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(1000); T = t[-1]
w = np.linspace(0.4, 3.6, 8000)
W = np.array([np.sum(np.cos(v*t)) for v in w])
plt.figure(figsize=(6, 4)); plt.plot(w, W, lw=0.7)
for q in range(2, 37):
    ps = [p for p in qmnt.primes_upto(q) if q % p == 0]
    if len(ps) == 1:
        plt.plot(np.log(q), -(T/(2*np.pi))*np.log(ps[0])/np.sqrt(q), 'ko', ms=3)
plt.xlabel(r"$\omega$ (rad per unit $t$)"); plt.ylabel(r"$\sum_n\cos(\omega t_n)$")
plt.savefig("fourier-transform_landau.pdf", bbox_inches="tight")
