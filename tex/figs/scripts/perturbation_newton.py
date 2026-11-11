import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb=1.0,60
n=np.arange(1,Nb+1); E0=(n*np.pi/(2*L))**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
C1=-0.5*((D==2)*1.0-(S==2)*1.0)
def ground(c):
    w,v=np.linalg.eigh(np.diag(E0)+c*C1); return w[0],v[:,0]@C1@v[:,0]
cs=np.linspace(0,1.4,200); Es=[ground(c)[0] for c in cs]
E00,A0=ground(0.0); c1=(3-E00)/A0
E1,A1=ground(c1); c2=c1+(3-E1)/A1
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(cs,Es,'k-',label=r'exact $E_1(c)$')
ax.axhline(3,color='gray',lw=0.8,label='target 3')
ax.plot(cs,E00+A0*cs,'b--',label='tangent at c = 0')
ax.plot(cs,E1+A1*(cs-c1),'r:',lw=2,label='tangent at $c_1$')
ax.plot([c1,c2],[3,3],'ko',ms=4)
ax.text(c1,2.9,'$c_1$',ha='right'); ax.text(c2,2.9,'$c_2$',ha='left')
ax.set_xlim(0,1.4); ax.set_ylim(2.4,3.3)
ax.set_xlabel(r'coefficient $c$ (energy, units $\hbar=2m=1$)')
ax.set_ylabel(r'ground-state energy $E_1$')
ax.legend(fontsize=8)
plt.savefig("perturbation_newton.pdf", bbox_inches="tight")
