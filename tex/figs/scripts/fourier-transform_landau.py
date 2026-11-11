import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(1000)
T = t[-1]
w = np.linspace(0.4, 3.6, 4000)
S = np.cos(np.outer(w, t)).sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(w, S, lw=0.8)
qmax = np.exp(3.6)
for p in qmnt.primes_upto(int(qmax)):
    q = p
    while q <= qmax:
        if np.log(q) >= 0.4:
            ax.plot(np.log(q), -(T/(2*np.pi))*np.log(p)/np.sqrt(q), 'ko', ms=4)
        q *= p
ax.set_xlabel('omega (rad per unit t)'); ax.set_ylabel('sum of cos(omega t_n), n <= 1000')
plt.savefig("fourier-transform_landau.pdf", bbox_inches="tight")
