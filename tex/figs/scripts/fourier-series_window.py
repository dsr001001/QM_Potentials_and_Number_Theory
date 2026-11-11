import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = lambda t: np.sin(t*np.log(2)) + np.sin(t*np.log(3))
T = 40.0
t = np.linspace(0, 120, 6000)
plt.figure(figsize=(6, 4))
plt.axvspan(0, T, color='0.9')
plt.plot(t, f(t), 'k', lw=1, label='true function')
plt.plot(t, f(np.mod(t, T)), 'C3--', lw=1, label='window series (repeats)')
plt.xlabel('t (dimensionless)'); plt.ylabel('f(t)'); plt.legend(fontsize=8)
plt.savefig("fourier-series_window.pdf", bbox_inches="tight")
