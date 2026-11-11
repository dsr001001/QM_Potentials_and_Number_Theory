import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x=np.linspace(-4,4,800); V=0.5*x**2+0.3*x; E=4.0
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,V,'k',label='V(x)'); ax.axhline(E,color='b',lw=1)
r=np.roots([0.5,0.3,-E]); x1,x2=sorted(r.real)
ax.axvspan(-4,x1,color='0.85'); ax.axvspan(x2,4,color='0.85')
xa=np.linspace(x1+0.02,x2-0.02,600); k=np.sqrt(np.maximum(E-0.5*xa**2-0.3*xa,1e-9))
ph=np.concatenate([[0],np.cumsum(k[1:]*np.diff(xa))])*3
ax.plot(xa,E+0.6*np.cos(ph)/np.sqrt(k),'r',lw=0.8)
ax.text(x1,E+0.3,'$x_1$'); ax.text(x2,E+0.3,'$x_2$')
ax.set_xlabel('x (units $\\hbar=2m=1$)'); ax.set_ylabel('energy'); ax.set_ylim(0,8)
plt.savefig("wkb_turning.pdf", bbox_inches="tight")
