import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.linspace(-8, 8, 8001)
box = np.abs(np.sinc(d))
hann = np.abs(0.5*np.sinc(d) + 0.25*np.sinc(d-1) + 0.25*np.sinc(d+1))/0.5
db = lambda a: 20*np.log10(np.maximum(a, 1e-5))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d, db(box), label='box (rectangular)')
ax.plot(d, db(hann), '--', label='Hann')
ax.set_ylim(-100, 5)
ax.set_xlabel('offset from the line (bins), delta')
ax.set_ylabel('amplitude relative to peak (dB)')
ax.legend()
plt.savefig("fourier-transform_leakage.pdf", bbox_inches="tight")
