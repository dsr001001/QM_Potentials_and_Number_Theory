import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 12, 3000); j = np.arange(13)
plt.figure(figsize=(6, 4))
plt.plot(t, np.cos(0.5*t), label=r"$\cos(0.5t)$")
plt.plot(t, np.cos((2*np.pi-0.5)*t), lw=0.7, label=r"$\cos(5.783t)$")
plt.plot(j, np.cos(0.5*j), 'ko', label="samples")
plt.xlabel(r"$t$"); plt.ylabel("signal"); plt.legend(loc="lower left")
plt.savefig("fourier-transform_aliasing.pdf", bbox_inches="tight")
