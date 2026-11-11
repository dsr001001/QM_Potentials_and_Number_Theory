import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
xs = np.linspace(0, 4, 400)
fig, ax = plt.subplots(figsize=(6, 4))
for E, st in [(0.9, '-'), (1.0, '-'), (1.1, '--')]:
    sol = solve_ivp(lambda x, y: [y[1], (x**2 - E)*y[0]], (0, 4), [1.0, 0.0],
                    t_eval=xs, rtol=1e-11, atol=1e-13)
    ax.plot(sol.t, sol.y[0], st, lw=2 if E == 1.0 else 1.2, label=f"E = {E}")
ax.axvline(1, ls=':', color='k')
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylim(-2, 3)
ax.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"$\psi(x)$ (arbitrary units)")
ax.legend()
plt.savefig("bound-states_shooting.pdf", bbox_inches="tight")
