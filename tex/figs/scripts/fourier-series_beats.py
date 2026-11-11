import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t = np.linspace(0, 120, 6000)
f = np.sin(t*np.log(11)) + np.sin(t*np.log(13))
env = 2*np.abs(np.cos(t*np.log(13/11)/2))
plt.figure(figsize=(6, 4))
plt.plot(t, f, color='0.6', lw=0.8)
plt.plot(t, env, 'k'); plt.plot(t, -env, 'k')
plt.xlabel('t (dimensionless)'); plt.ylabel(r'$\sin(t\log 11)+\sin(t\log 13)$')
plt.savefig("fourier-series_beats.pdf", bbox_inches="tight")
