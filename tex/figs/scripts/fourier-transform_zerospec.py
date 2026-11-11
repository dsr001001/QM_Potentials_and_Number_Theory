import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(1000)
n = np.arange(1, 1001)
y = n - 0.5 - qmnt.Nsm(t); y = y - y.mean()
w = np.linspace(0.3, 3.0, 6000)
P = np.abs(np.exp(-1j*np.outer(w, t)) @ y)**2/len(t)
qs = []
for p in qmnt.primes_upto(36):
    q = p
    while q <= 36:
        qs.append(q); q *= p
qs = sorted(qs)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(w, P, lw=0.8)
for q in qs:
    ax.axvline(np.log(q), color='gray', ls='--', lw=0.5)
for q in (2, 3, 4, 5, 7, 8, 9):
    ax.text(np.log(q), P.max()*1.02, str(q), ha='center', fontsize=7)
ax.set_xlabel('omega (rad per unit t)'); ax.set_ylabel('periodogram P')
plt.savefig("fourier-transform_zerospec.pdf", bbox_inches="tight")
