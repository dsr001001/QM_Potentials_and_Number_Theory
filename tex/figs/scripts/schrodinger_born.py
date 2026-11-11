import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
a = 1.0
x = np.linspace(-4, 4, 1601)
rho = np.exp(-2*np.abs(x)/a)/a
ax.plot(x, rho, "k")
m = np.abs(x) < a
ax.fill_between(x[m], 0, rho[m], alpha=0.35, label="P(|x|<a) = 1 - e^-2 = 0.865")
ax.set_xlabel("x / a")
ax.set_ylabel("|psi|^2  (units 1/a)")
ax.legend()
plt.savefig("schrodinger_born.pdf", bbox_inches="tight")
