import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ai_zeros
a,ap,_,_=ai_zeros(30); lev=np.sort(np.concatenate([-a,-ap]))
E=np.linspace(0,12,3000)
N=np.searchsorted(lev,E,side='right')
sm=4/(3*np.pi)*E**1.5
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(E,N,'k-',drawstyle='steps-post',label='exact count $\\mathcal{N}(E)$')
ax.plot(E,sm,'b-',label='$\\Phi/\\pi$')
ax.plot(E,sm+0.5,'r--',label='$\\Phi/\\pi+1/2$')
ax.set_xlabel('energy $E$ (units $\\hbar=2m=1$)'); ax.set_ylabel('number of levels')
ax.legend(fontsize=8)
plt.savefig("wkb_staircase.pdf", bbox_inches="tight")
