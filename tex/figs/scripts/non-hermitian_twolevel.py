import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=np.linspace(0,1,400); d=np.sqrt((1-4*a**2).astype(complex))
lp,lm=0.5+d/2,0.5-d/2
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3))
ax1.plot(a,lp.real,"b"); ax1.plot(a,lm.real,"r")
ax1.plot(a,a**2,"r--"); ax1.plot(a,1-a**2,"b--"); ax1.axvline(0.5,ls=":",c="k")
ax1.set_xlabel("coupling a"); ax1.set_ylabel("Re eigenvalue (energy units)")
ax2.plot(a,lp.imag,"b"); ax2.plot(a,lm.imag,"r"); ax2.axvline(0.5,ls=":",c="k")
ax2.set_xlabel("coupling a"); ax2.set_ylabel("Im eigenvalue (energy units)")
plt.tight_layout(); plt.savefig("non-hermitian_twolevel.pdf",bbox_inches="tight")
