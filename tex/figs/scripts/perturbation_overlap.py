import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L=1.0; x=np.linspace(-L,L,800)
rho=np.sin(3*np.pi*(x+L)/(2*L))**2/L
fig,ax=plt.subplots(1,2,figsize=(6.5,3),sharey=True)
for a,q in zip(ax,[3,2]):
    cq=np.cos(q*np.pi*x/L); prod=rho*cq
    a.plot(x,rho,label=r"$|\psi_3|^2$"); a.plot(x,cq,label=rf"$\cos({q}\pi x/L)$")
    a.fill_between(x,prod,alpha=0.3,color="gray",label="product")
    a.set_title(f"q={q}: integral={np.trapezoid(prod,x):.2f}"); a.set_xlabel("x (L=1)")
ax[0].set_ylabel("value"); ax[0].legend(fontsize=7)
plt.savefig("perturbation_overlap.pdf",bbox_inches="tight")
