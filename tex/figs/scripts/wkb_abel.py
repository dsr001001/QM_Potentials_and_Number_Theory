import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import quad
import qmnt
V0=qmnt.V0
def xab(W):
    a=np.sqrt(W-V0); return quad(lambda u: np.log((W-u*u)/(2*np.pi)),0,a)[0]/np.pi
Ws=np.linspace(V0+1,1000,25); Wc=np.linspace(V0,1000,500)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(Wc,[qmnt.wu_sprung_x(w) for w in Wc],'k',label='Eq. (2)')
ax.plot(Ws,[xab(w) for w in Ws],'ro',label='numerical Abel')
ax.set_xlabel('W (units $\\hbar=2m=1$)'); ax.set_ylabel('x(W)'); ax.legend()
plt.savefig("wkb_abel.pdf", bbox_inches="tight")
