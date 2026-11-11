import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial
x = np.linspace(-4.5, 4.5, 800)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x**2, 'k', lw=1)
for n in range(1, 6):
    En = 2*n - 1
    norm = 1/np.sqrt(2.0**(n - 1)*factorial(n - 1)*np.sqrt(np.pi))
    psi = norm*eval_hermite(n - 1, x)*np.exp(-x**2/2)
    ax.axhline(En, color='gray', lw=0.5)
    ax.plot(x, En + 1.5*psi, lw=1.5)
    ax.plot([-np.sqrt(En), np.sqrt(En)], [En, En], 'ko', ms=4)
ax.set_ylim(0, 11)
ax.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"energy; $\psi_n$ on baselines (arbitrary units)")
plt.savefig("bound-states_oscillator.pdf", bbox_inches="tight")
