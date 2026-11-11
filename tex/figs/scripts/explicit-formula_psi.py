import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

x = np.linspace(1.5, 40, 4000)
pr = qmnt.primes_upto(40)
psi = np.zeros_like(x)
for p in pr:
    k = p
    while k <= 40:
        psi += np.log(p) * (x >= k)
        k *= p
z = qmnt.zeros(100)
def approx(K):
    rho = 0.5 + 1j * z[:K]
    lx = np.log(x)[:, None]
    waves = 2 * np.real(np.exp(rho[None, :] * lx) / rho[None, :]).sum(axis=1)
    return x - waves - np.log(2 * np.pi) - 0.5 * np.log(1 - x ** -2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, psi, "k-", lw=1.2, label=r"$\psi(x)$")
ax.plot(x, approx(10), "--", lw=1, label="10 zero pairs")
ax.plot(x, approx(100), "-", lw=1, alpha=0.8, label="100 zero pairs")
ax.set_xlabel("x (dimensionless)")
ax.set_ylabel(r"$\psi(x)$ (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("explicit-formula_psi.pdf", bbox_inches="tight")
