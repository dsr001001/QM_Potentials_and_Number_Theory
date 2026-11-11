import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ai_zeros
a,ap,_,_=ai_zeros(20); lev=np.sort(np.concatenate([-a,-ap]))
E=np.linspace(0,12,1000); N=np.array([(lev<=e).sum() for e in E])
fig,ax=plt.subplots(figsize=(6,4)); ax.step(E,N,'k',where='post',label='exact count')
ax.plot(E,4/(3*np.pi)*E**1.5,'b',label='$\\Phi/\\pi$'); ax.plot(E,4/(3*np.pi)*E**1.5+0.5,'r--',label='$\\Phi/\\pi+1/2$')
ax.set_xlabel('E (units $\\hbar=2m=1$)'); ax.set_ylabel('number of levels'); ax.legend()
plt.savefig("wkb_staircase.pdf", bbox_inches="tight")
