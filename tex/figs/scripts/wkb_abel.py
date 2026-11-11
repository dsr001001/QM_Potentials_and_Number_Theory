import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import quad
import qmnt
V0=qmnt.V0
Wp=np.linspace(V0+1,1000,25)
xp=[quad(lambda u:np.log((W-u*u)/(2*np.pi)),0,np.sqrt(W-V0))[0]/np.pi for W in Wp]
Wc=np.linspace(V0,1000,400)
xc=np.array([qmnt.wu_sprung_x(W) for W in Wc])
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(Wc,xc,'k-',label='closed form, paper Eq. (2)')
ax.plot(Wp,xp,'ro',ms=4,label='numerical Abel inversion')
ax.set_xlabel('height $W$ (units $\\hbar=2m=1$)'); ax.set_ylabel('turning point $x(W)$')
ax.legend(fontsize=8)
plt.savefig("wkb_abel.pdf", bbox_inches="tight")
