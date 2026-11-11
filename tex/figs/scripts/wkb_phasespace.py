import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,4)); th=np.linspace(0,2*np.pi,400)
for j,E in enumerate([8,6,4,2]):
    ax.fill(np.sqrt(E)*np.cos(th),np.sqrt(E)*np.sin(th),color=['0.7','0.9'][j%2])
ax.fill(np.sqrt(2)*np.cos(th),np.sqrt(2)*np.sin(th),color='0.9')
for E in [1,3,5,7]: ax.plot(np.sqrt(E)*np.cos(th),np.sqrt(E)*np.sin(th),'k--',lw=1)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('p (units $\\hbar=2m=1$)')
plt.savefig("wkb_phasespace.pdf", bbox_inches="tight")
