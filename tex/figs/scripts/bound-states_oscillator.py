import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial, pi, sqrt

x = np.linspace(-4.2, 4.2, 800)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x**2, color="k", lw=1.2, label=r"$V=x^2$")
for n in range(1, 6):
    j = n - 1
    c = np.zeros(j + 1); c[j] = 1.0
    norm = 1.0/sqrt(2**j*factorial(j)*sqrt(pi))
    psi = norm*hermval(x, c)*np.exp(-x**2/2)
    En = 2*n - 1
    ax.axhline(En, color="0.85", lw=0.8)
    ax.plot(x, En + 1.5*psi, lw=1.5)
    xt = np.sqrt(En)
    ax.plot([-xt, xt], [En, En], "o", color="C3", ms=4)
ax.set_ylim(-0.5, 11.5)
ax.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"energy (units $\hbar=2m=1$); $E_n+1.5\,\psi_n$")
ax.set_title("Harmonic oscillator: levels $E_n=2n-1$")
ax.legend(loc="upper center")
plt.savefig("bound-states_oscillator.pdf", bbox_inches="tight")
