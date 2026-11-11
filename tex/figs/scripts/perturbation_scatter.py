import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pi=np.pi; L=44.883; Nb=300; K=20; nl=100
u=(pi/(2*L))**2; n=np.arange(1,Nb+1); E0=u*n**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
def cm(q): return 0.5*(-1)**q*((D==2*q)*1.0-(S==2*q)*1.0)
Cs=[np.eye(Nb)]+[cm(q) for q in range(1,K)]
z=np.random.default_rng(0).standard_normal(K)
sp=u*(2*n[:nl]+1)
fig,axs=plt.subplots(1,2,figsize=(6.5,3))
for ax,sig in zip(axs,[0.2,20.0]):
    c=sig*u*z
    H=np.diag(E0)+sum(ck*Ck for ck,Ck in zip(c,Cs))
    Ex=np.linalg.eigvalsh(H)[:nl]
    fo=np.array([sum(c[k]*Cs[k][i,i] for k in range(K)) for i in range(nl)])
    y=(Ex-E0[:nl])/sp; f=fo/sp
    sc=ax.scatter(f,y,c=n[:nl],cmap='viridis',s=10)
    lo=min(f.min(),y.min()); hi=max(f.max(),y.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=1)
    ax.set_xlabel('first-order shift (local spacings)')
    ax.set_ylabel('exact shift (local spacings)')
    ax.set_title('std = %g (pi/2L)^2' % sig, fontsize=9)
fig.colorbar(sc,ax=axs,label='level n')
plt.savefig('perturbation_scatter.pdf',bbox_inches='tight')
