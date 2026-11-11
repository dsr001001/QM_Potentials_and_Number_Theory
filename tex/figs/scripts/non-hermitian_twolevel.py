import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=np.linspace(0,1,401)
r=np.sqrt((1-4*a**2).astype(complex))
lp=0.5+0.5*r; lm=0.5-0.5*r
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.5,3))
ax1.plot(a,lp.real,'b',label='exact'); ax1.plot(a,lm.real,'b')
ax1.plot(a,1-a**2,'r--',label='2nd order'); ax1.plot(a,a**2,'r--')
ax1.axvline(0.5,color='gray',ls=':')
ax1.set_xlabel('coupling a (energy units)'); ax1.set_ylabel(r'Re $\lambda$ (energy units)'); ax1.set_title('(a)'); ax1.legend(fontsize=8)
ax2.plot(a,lp.imag,'b'); ax2.plot(a,lm.imag,'b')
ax2.axvline(0.5,color='gray',ls=':')
ax2.set_xlabel('coupling a (energy units)'); ax2.set_ylabel(r'Im $\lambda$ (energy units)'); ax2.set_title('(b)')
plt.tight_layout()
plt.savefig("non-hermitian_twolevel.pdf", bbox_inches="tight")
