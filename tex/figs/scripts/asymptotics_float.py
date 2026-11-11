import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x=np.linspace(1,16,10000); rel=(x.astype(np.float32).astype(np.float64)-x)/x
e=np.finfo(np.float32).eps
plt.figure(figsize=(6,4)); plt.plot(x,rel,',k')
plt.axhline(e/2,ls='--'); plt.axhline(-e/2,ls='--')
for p in [2,4,8]: plt.axvline(p,ls=':',color='gray')
plt.xlabel("x (dimensionless)"); plt.ylabel("relative rounding error, float32")
plt.savefig("asymptotics_float.pdf", bbox_inches="tight")
