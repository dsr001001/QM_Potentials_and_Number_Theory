import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4))
th=np.linspace(0,2*np.pi,400)
for i,Eb in enumerate([8,6,4,2]):
    r=np.sqrt(Eb); col='0.75' if i%2==0 else 'white'
    ax.fill(r*np.cos(th),r*np.sin(th),color=col,ec='k',lw=0.5)
for En in [1,3,5,7]:
    r=np.sqrt(En); ax.plot(r*np.cos(th),r*np.sin(th),'b--',lw=1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ (units $\\hbar=2m=1$)'); ax.set_ylabel('$p$')
ax.set_title('rings of area $2\\pi$; dashed: $E=1,3,5,7$',fontsize=9)
plt.savefig("wkb_phasespace.pdf", bbox_inches="tight")
