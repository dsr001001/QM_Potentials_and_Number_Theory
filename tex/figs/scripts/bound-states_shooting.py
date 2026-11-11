import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rhs(x, y, E):
    return [y[1], (x**2 - E)*y[0]]

xs = np.linspace(0, 3.6, 500)
fig, ax = plt.subplots(figsize=(6, 4))
for E, style in [(0.9, "-"), (1.0, "-"), (1.1, "-")]:
    sol = solve_ivp(rhs, (0, 3.6), [1.0, 0.0], args=(E,), t_eval=xs,
                    rtol=1e-11, atol=1e-13)
    ax.plot(sol.t, sol.y[0], style, lw=1.6, label="E = %.1f" % E)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(1.0, color="0.5", ls=":", lw=1)
ax.set_ylim(-1.5, 1.8)
ax.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
ax.set_ylabel(r"$\psi(x)$ (with $\psi(0)=1$)")
ax.set_title(r"Shooting in $V=x^2$: only $E=1$ decays")
ax.legend()
plt.savefig("bound-states_shooting.pdf", bbox_inches="tight")
