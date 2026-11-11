import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
w0 = np.log(2)
w = np.linspace(0, 1.4, 2000)
def g(w, w0, T):
    x = (w-w0)*T/2
    return np.sinc(x/np.pi)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
for T, ls in [(20, '-'), (40, '--')]:
    ax[0].plot(w, g(w, w0, T), ls, label=f"T={T}")
ax[0].axvline(w0, color='gray', lw=0.5)
ax[0].set_xlabel(r"$\omega$ (rad per unit $t$)"); ax[0].set_ylabel(r"$\hat g/T$"); ax[0].legend()
w = np.linspace(1.6, 2.4, 2000)
for T, ls in [(30, '-'), (100, '--')]:
    s = np.abs(g(w, np.log(7), T) + g(w, np.log(8), T))
    ax[1].plot(w, s/s.max(), ls, label=f"T={T}")
for q in (7, 8): ax[1].axvline(np.log(q), color='gray', lw=0.5)
ax[1].set_xlabel(r"$\omega$ (rad per unit $t$)"); ax[1].set_ylabel("normalized amplitude"); ax[1].legend()
plt.tight_layout()
plt.savefig("fourier-transform_sinc.pdf", bbox_inches="tight")
