import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb,K=44.883,300,20; n=np.arange(1,Nb+1); u=(np.pi/2/L)**2; E0=u*n**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
def cm(q): return np.eye(Nb) if q==0 else 0.5*(-1)**q*((D==2*q)*1.0-(S==2*q)*1.0)
mats=[cm(q) for q in range(K)]; rng=np.random.default_rng(0); z=rng.standard_normal(K)
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
for a,s in zip(ax,[0.2,20]):
    c=s*u*z; W=sum(ci*m for ci,m in zip(c,mats))
    Ee=np.linalg.eigvalsh(np.diag(E0)+W)[:100]; p1=np.diag(W)[:100]
    g=u*2*n[:100]
    a.scatter(p1/g,(Ee-E0[:100])/g,c=n[:100],s=8)
    lim=np.max(np.abs(p1/g))*1.2+1e-9; a.plot([-lim,lim],[-lim,lim],"k--")
    a.set_xlabel("first-order shift (spacings)"); a.set_title(f"sd = {s}"+r"$(\pi/2L)^2$")
ax[0].set_ylabel("exact shift (spacings)")
plt.savefig("perturbation_scatter.pdf",bbox_inches="tight")
