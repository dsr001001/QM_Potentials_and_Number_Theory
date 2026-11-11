import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import gammaln
n=np.arange(1,201).astype(float); ex=gammaln(n+1)
a=n*np.log(n)-n; b=a+0.5*np.log(2*np.pi*n); c=b+1/(12*n)
plt.figure(figsize=(6,4))
plt.loglog(n,abs(ex-a),label="n log n - n")
plt.loglog(n,abs(ex-b),label="+ (1/2)log(2 pi n)")
plt.loglog(n[:60],abs(ex-c)[:60],label="+ 1/(12n)")
plt.loglog(n,1/(12*n),'k--',lw=0.8); plt.loglog(n,1/(360*n**3),'k:',lw=0.8)
plt.xlabel("n (dimensionless)"); plt.ylabel("absolute error of log n!"); plt.legend()
plt.savefig("asymptotics_stirling.pdf", bbox_inches="tight")
