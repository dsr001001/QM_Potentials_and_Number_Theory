import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

a = 1.0
x = np.linspace(-4, 4, 801)
rho = np.exp(-2*np.abs(x)/a)/a
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, rho, "k-")
m = np.abs(x) < a
ax.fill_between(x[m], 0, rho[m], color="tab:blue", alpha=0.4)
P = 1 - np.exp(-2)
ax.text(0, 0.35, f"P(|x|<a) = {P:.3f}", ha="center", fontsize=10)
ax.set_xlabel(r"position $x$ (units of $a$)")
ax.set_ylabel(r"$|\psi|^2$ (units of $1/a$)")
plt.savefig("schrodinger_born.pdf", bbox_inches="tight")
