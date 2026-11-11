import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = np.linspace(-2, 2, 801)
Em = lambda c: 2 - np.sqrt(1 + c**2)
plt.figure(figsize=(6, 4))
plt.plot(c, np.abs(0.5 - Em(c)), "k-")
ci = [0.5]
for _ in range(4):
    cc = ci[-1]; r = 0.5 - Em(cc); J = -cc/np.sqrt(1 + cc**2)
    ci.append(cc + r/J)
ci = np.array(ci)
plt.plot(ci, np.abs(0.5 - Em(ci)), "o", color="C3")
for i, cc in enumerate(ci[:3]):
    plt.annotate(str(i), (cc, abs(0.5 - Em(cc))), textcoords="offset points", xytext=(4, 6))
plt.axvline(np.sqrt(1.25), ls=":", c="gray"); plt.axvline(-np.sqrt(1.25), ls=":", c="gray")
plt.xlabel("coefficient c (dimensionless)"); plt.ylabel("|residual| (dimensionless)")
plt.savefig("fitting_landscape.pdf", bbox_inches="tight")
