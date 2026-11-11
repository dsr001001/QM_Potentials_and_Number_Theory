import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = 1.0
x = np.linspace(-L, L, 500)
fig, ax = plt.subplots(figsize=(6, 4))
for n in range(1, 6):
    psi = np.sin(n*np.pi*(x + L)/(2*L))
    ax.axhline(n**2, color='gray', lw=0.6)
    ax.plot(x, n**2 + 1.2*psi, lw=1.5)
    ax.text(1.05, n**2, f"n={n}", va='center')
ax.plot([-L, -L], [0, 28], 'k', lw=4)
ax.plot([L, L], [0, 28], 'k', lw=4)
ax.set_xlim(-1.15, 1.3)
ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"$E_n/E_1$ (baselines); $\psi_n$ (arbitrary units)")
plt.savefig("bound-states_box.pdf", bbox_inches="tight")
