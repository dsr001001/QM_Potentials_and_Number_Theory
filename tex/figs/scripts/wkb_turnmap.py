import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t=qmnt.zeros(300)
Vs=np.linspace(qmnt.V0,1000,400)
xs=np.array([qmnt.wu_sprung_x(v) for v in Vs])
x1=qmnt.wu_sprung_x(t[200]); x2=qmnt.wu_sprung_x(t[299])
fig,ax=plt.subplots(figsize=(6,4))
ax.axvspan(x1,x2,color='0.85')
ax.plot(xs,Vs,'k-',label='Wu--Sprung $V(x)$')
ax.axhline(t[200],color='b',ls='--',lw=1,label='$t_{201}$')
ax.axhline(t[299],color='r',ls='--',lw=1,label='$t_{300}$')
ax.set_xlabel('$x$ (units $\\hbar=2m=1$)'); ax.set_ylabel('$V$, $E$')
ax.legend(fontsize=8,loc='upper left')
plt.savefig("wkb_turnmap.pdf", bbox_inches="tight")
