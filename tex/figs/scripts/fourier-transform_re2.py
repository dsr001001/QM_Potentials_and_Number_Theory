import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
tz = qmnt.zeros(400)
f = lambda t: -sum(np.sin(m*t*np.log(2))/(m*2**(m/2)) for m in (1,2,3))/np.pi
w = np.linspace(0.3, 2.5, 3000)
def P(t, y):
    y = y - y.mean()
    return np.array([abs(np.sum(y*np.exp(-1j*v*t)))**2/len(t) for v in w])
tu = np.arange(tz[0], tz[-1], 0.5)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
ax[0].semilogy(w, P(tu, f(tu)) + 1e-8)
for m in (1,2,3):
    A = 1/(np.pi*m*2**(m/2)); ax[0].plot(m*np.log(2), len(tu)*A**2/4, 'ko')
ax[1].semilogy(w, P(tz, f(tz)) + 1e-8)
for a in ax:
    for m in (1,2,3): a.axvline(m*np.log(2), color='gray', lw=0.5)
    a.set_xlabel(r"$\omega$ (rad per unit $t$)")
ax[0].set_ylabel(r"$P(\omega)$"); ax[0].set_title("uniform"); ax[1].set_title("at zeros")
plt.tight_layout()
plt.savefig("fourier-transform_re2.pdf", bbox_inches="tight")
