import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
l2, l3 = np.log(2), np.log(3)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
t = np.linspace(0, 200, 8000)
ax[0].plot(t, np.sin(t*l2)+np.sin(t*l3), 'k', lw=0.7)
t0 = 20.5*np.pi/l2
ax[0].plot([t0], [np.sin(t0*l2)+np.sin(t0*l3)], 'ro')
ax[0].set_xlabel('t'); ax[0].set_ylabel('f(t)')
tau = np.linspace(0.5, 600, 60000)
D = 2*np.abs(np.sin(tau*l2/2)) + 2*np.abs(np.sin(tau*l3/2))
ax[1].semilogy(tau, D, 'k', lw=0.5)
ax[1].set_xlabel(r'shift $\tau$'); ax[1].set_ylabel(r'$D(\tau)$'); ax[1].set_ylim(1e-2, 5)
plt.tight_layout()
plt.savefig("fourier-series_quasiperiodic.pdf", bbox_inches="tight")
