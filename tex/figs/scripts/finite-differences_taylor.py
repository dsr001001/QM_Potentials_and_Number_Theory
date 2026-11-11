import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h = np.logspace(-6, 0, 61)
D = (np.sin(1+h)-2*np.sin(1)+np.sin(1-h))/h**2
err = np.abs(D+np.sin(1))
fig, ax = plt.subplots(figsize=(6,4))
ax.loglog(h, err, 'o', ms=4, label='actual error (double precision)')
ax.loglog(h, h**2*np.sin(1)/12, 'k-', label=r'$h^2\sin(1)/12$')
ax.set_xlabel('step h (dimensionless)')
ax.set_ylabel('absolute error of second difference')
ax.legend()
plt.savefig("finite-differences_taylor.pdf", bbox_inches="tight")
