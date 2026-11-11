import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
x = np.linspace(1.5, 50, 3000)
psi = np.zeros_like(x)
for p in qmnt.primes_upto(50):
    q = p
    while q <= 50:
        psi += np.log(p)*(x >= q); q *= p
zz = np.asarray(qmnt.zeros(100))
def trunc(K):
    r = 0.5 + 1j*zz[:K]
    s = np.array([np.sum(np.real(np.exp(r*np.log(v))/r)) for v in x])
    return x - 2*s - np.log(2*np.pi) - 0.5*np.log(1 - x**-2.0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, psi, 'k', lw=1, label=r"$\psi(x)$")
ax.plot(x, trunc(10), lw=1, label="10 zero pairs")
ax.plot(x, trunc(100), lw=1, label="100 zero pairs")
ax.set_xlabel("x (dimensionless)"); ax.set_ylabel(r"$\psi(x)$ (dimensionless)")
ax.legend()
plt.savefig("explicit-formula_psi.pdf", bbox_inches="tight")
