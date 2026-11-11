import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import airy, ai_zeros
a,ap,_,_=ai_zeros(5); E=-a[4]
x=np.linspace(0,E+3,3000)
ex=airy(x-E)[0]
D=1/(2*np.sqrt(np.pi))
xl=x[x<E-0.08]; k=np.sqrt(E-xl)
wl=2*D/np.sqrt(k)*np.cos(2/3*(E-xl)**1.5-np.pi/4)
xr=x[x>E+0.08]; kap=np.sqrt(xr-E)
wr=D/np.sqrt(kap)*np.exp(-2/3*(xr-E)**1.5)
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(x,ex,'k-',label='exact $\\mathrm{Ai}(x-E)$')
ax.plot(xl,wl,'r--',label='WKB')
ax.plot(xr,wr,'r--')
ax.axvline(E,color='gray',ls=':')
ax.set_ylim(-0.9,0.9)
ax.set_xlabel('$x$ (units $\\hbar=2m=1$)'); ax.set_ylabel('$\\psi$ (arbitrary normalization)')
ax.legend(fontsize=8)
plt.savefig("wkb_compare.pdf", bbox_inches="tight")
