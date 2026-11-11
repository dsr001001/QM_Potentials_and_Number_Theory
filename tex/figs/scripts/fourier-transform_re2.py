import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
A = [1/(np.pi*m*2**(m/2)) for m in (1, 2, 3)]
f = lambda tt: -sum(A[m-1]*np.sin(m*tt*np.log(2)) for m in (1, 2, 3))
tz = qmnt.zeros(400)
tu = np.arange(tz[0], tz[-1], 0.5)
w = np.linspace(0.3, 2.5, 3000)
def P(tt):
    y = f(tt); y = y - y.mean()
    return np.abs(np.exp(-1j*np.outer(w, tt)) @ y)**2/len(tt)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for a_, tt, title in [(ax[0], tu, 'uniform samples'), (ax[1], tz, 'sampled at zeros')]:
    a_.semilogy(w, P(tt), lw=0.8)
    for m in (1, 2, 3):
        a_.axvline(m*np.log(2), color='gray', ls=':', lw=0.7)
    a_.set_title(title, fontsize=9); a_.set_xlabel('omega (rad per unit t)')
for m in (1, 2, 3):
    ax[0].plot(m*np.log(2), len(tu)*A[m-1]**2/4, 'ko', ms=4)
ax[0].set_ylabel('periodogram P')
ax[0].set_ylim(1e-6, 1e2)
plt.tight_layout()
plt.savefig("fourier-transform_re2.pdf", bbox_inches="tight")
