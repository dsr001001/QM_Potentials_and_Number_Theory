import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.linspace(0, 1.2, 400)
ex = 0.5*(1-np.sqrt(np.clip(1-4*x**2, 0, None)))
wt = []
for xi in x:
    w, v = np.linalg.eig(np.array([[0, 1j*xi],[1j*xi, 1.0]]))
    i = np.argmin(w.real + 1e-9*np.abs(w.imag)); u = v[:, i]
    wt.append(abs(u[0])**2/np.sum(abs(u)**2))
fig, ax = plt.subplots(1, 2, figsize=(6.5,3))
ax[0].plot(x, ex, label="exact"); ax[0].plot(x, x**2, "--", label="second order")
ax[0].axvline(0.5, c="gray", ls=":"); ax[0].set_xlabel(r"$x=a/\Delta$"); ax[0].set_ylabel(r"$\Re\,\delta E_1/\Delta$"); ax[0].legend()
ax[1].plot(x, wt); ax[1].axvline(0.5, c="gray", ls=":")
ax[1].set_xlabel(r"$x=a/\Delta$"); ax[1].set_ylabel("weight of state 1")
plt.tight_layout()
plt.savefig("paper-spurious-floor_twolevel.pdf", bbox_inches="tight")
