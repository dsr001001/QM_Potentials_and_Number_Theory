import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Em = lambda c: 2 - np.sqrt(1 + c**2)
Jf = lambda c: -c/np.sqrt(1 + c**2)
c = np.linspace(-2, 2, 801)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(c, np.abs(0.5 - Em(c)), 'b-', label="|0.5 - E_-(c)|")
cs = [0.5]
for _ in range(3):
    cc = cs[-1]; cs.append(cc + (0.5 - Em(cc))/Jf(cc))
cs = np.array(cs)
ax.plot(cs, np.abs(0.5 - Em(cs)), 'ro-', ms=5, lw=0.8, label="Gauss-Newton iterates from c = 0.5")
for i, ci in enumerate(cs):
    ax.annotate(str(i), (ci, abs(0.5 - Em(ci))), textcoords="offset points", xytext=(4, 6), fontsize=8)
ax.axvline(np.sqrt(1.25), color='gray', ls=':'); ax.axvline(-np.sqrt(1.25), color='gray', ls=':')
ax.set_xlabel("coefficient c (dimensionless)")
ax.set_ylabel("absolute residual (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("fitting_landscape.pdf", bbox_inches="tight")
