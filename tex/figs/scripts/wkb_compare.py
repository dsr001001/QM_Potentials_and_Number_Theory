import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import airy, ai_zeros
a,_,_,_=ai_zeros(5); E=-a[4]
x=np.linspace(0,E+3,1500); ex=airy(x-E)[0]
xw=x[x<E-0.05]; w=(1/np.sqrt(np.pi))*(E-xw)**-0.25*np.cos(2/3*(E-xw)**1.5-np.pi/4)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,ex,'k',label='exact Ai(x-E)')
ax.plot(xw,w,'r--',label='WKB'); ax.axvline(E,ls=':'); ax.set_ylim(-0.8,0.8)
ax.set_xlabel('x (units $\\hbar=2m=1$)'); ax.set_ylabel('$\\psi$ (arb.)'); ax.legend()
plt.savefig("wkb_compare.pdf", bbox_inches="tight")
