import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb=1.0,60
n=np.arange(1,Nb+1); E0=(n*np.pi/(2*L))**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
C1=-0.5*((D==2)*1.0-(S==2)*1.0)
cs=np.linspace(0,60,121)
ex=[np.linalg.eigvalsh(np.diag(E0)+c*C1)[0] for c in cs]
first=E0[0]+cs/2
second=first-cs**2*L**2/(8*np.pi**2)
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(cs,ex,'k-',label='exact')
ax.plot(cs,first,'b--',label='first order')
ax.plot(cs,second,'r:',lw=2,label='second order')
ax.set_xlabel(r'amplitude $c$ (energy, units $\hbar=2m=1$)')
ax.set_ylabel(r'ground-state energy $E_1$')
ax.legend()
plt.savefig("perturbation_validity.pdf", bbox_inches="tight")
