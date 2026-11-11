import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lam=np.linspace(-2,2,401); v=0.5
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(lam,lam,'k--',lw=1,label='uncoupled')
ax.plot(lam,-lam,'k--',lw=1)
r=np.sqrt(lam**2+v**2)
ax.plot(lam,r,'b-',label='exact, v = 0.5')
ax.plot(lam,-r,'b-')
ax.annotate('',xy=(0,v),xytext=(0,-v),arrowprops=dict(arrowstyle='<->',color='r'))
ax.text(0.08,0,'min gap 2|v| = 1',color='r',va='center')
ax.set_xlabel(r'$\lambda$ (energy, units $\hbar=2m=1$)')
ax.set_ylabel('eigenvalue (energy)')
ax.legend()
plt.savefig("perturbation_repulsion.pdf", bbox_inches="tight")
