import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb,K=44.883,300,20
n=np.arange(1,Nb+1); u=(np.pi/(2*L))**2; E0=u*n**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
cm=lambda q: 0.5*(-1)**q*((D==2*q)*1.0-(S==2*q)*1.0)
mats=[np.eye(Nb)]+[cm(q) for q in range(1,K)]
rng=np.random.default_rng(0)
z=rng.standard_normal(K)
nn=np.arange(1,101); spacing=u*2*nn
fig,axs=plt.subplots(1,2,figsize=(6.5,3))
for ax,s in zip(axs,[0.2,20.0]):
    c=s*u*z
    H=np.diag(E0)+sum(ck*Mk for ck,Mk in zip(c,mats))
    E=np.linalg.eigvalsh(H)[:100]
    exact=(E-E0[:100])/spacing
    pred=np.full(100,c[0])
    for q in range(1,K): pred[q-1]+=(-1)**(q+1)*c[q]/2
    pred=pred/spacing
    sc=ax.scatter(pred,exact,c=nn,s=8,cmap='viridis')
    lo=min(pred.min(),exact.min()); hi=max(pred.max(),exact.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=0.8)
    ax.set_title('std = %g (pi/2L)^2'%s)
    ax.set_xlabel('first-order shift (local spacings)')
axs[0].set_ylabel('exact shift (local spacings)')
fig.colorbar(sc,ax=axs,label='level n')
plt.savefig("perturbation_scatter.pdf", bbox_inches="tight")
