import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(-L, L, 600)
fig, ax = plt.subplots(figsize=(6, 4))
for n in range(1, 6):
    En = n**2  # energy in units of E1 = (pi/2L)^2
    psi = np.sin(n*np.pi*(x + L)/(2*L))/np.sqrt(L)
    ax.axhline(En, color="0.8", lw=0.8)
    ax.plot(x/L, En + 1.2*np.sqrt(L)*psi, lw=1.6, label="n = %d" % n)
ax.axvline(-1, color="k", lw=3)
ax.axvline(1, color="k", lw=3)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-0.5, 27.5)
ax.set_xlabel(r"$x/L$ (dimensionless)")
ax.set_ylabel(r"$E_n/E_1$ (baseline) and $E_n/E_1+1.2\sqrt{L}\,\psi_n$")
ax.legend(loc="upper center", fontsize=8, ncol=5)
ax.set_title("Infinite square well on $[-L,L]$")
plt.savefig("bound-states_box.pdf", bbox_inches="tight")
