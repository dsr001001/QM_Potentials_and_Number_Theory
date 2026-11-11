import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import lambertw
n=np.arange(2,10001).astype(float); y=(n-11/8)/np.e; W=lambertw(y).real
s=2*np.pi*(n-11/8)/W; g=2*np.pi/(1+W)
plt.figure(figsize=(6,4))
plt.semilogx(n,(2*np.pi*n/np.log(n))/s,'k',label="(2 pi n/log n)/s_n")
plt.semilogx(n,(2*np.pi/np.log(n))/g,'k--',label="(2 pi/log n)/gap_n")
plt.axhline(1,color='gray',lw=0.5); plt.xlabel("index n"); plt.ylabel("ratio (dimensionless)"); plt.legend()
plt.savefig("asymptotics_lambertw.pdf", bbox_inches="tight")
