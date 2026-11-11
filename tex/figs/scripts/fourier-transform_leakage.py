import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.linspace(-8, 8, 4001) + 1e-9
box = np.abs(np.sinc(d))
hann = np.abs(-np.sin(np.pi*d)/(2*np.pi*d*(d**2-1+1e-12)))/0.5
plt.figure(figsize=(6, 4))
plt.plot(d, 20*np.log10(box+1e-12), label="box")
plt.plot(d, 20*np.log10(hann+1e-12), '--', label="Hann")
plt.ylim(-80, 3); plt.xlabel(r"offset $\delta$ (bins)"); plt.ylabel("amplitude (dB)"); plt.legend()
plt.savefig("fourier-transform_leakage.pdf", bbox_inches="tight")
