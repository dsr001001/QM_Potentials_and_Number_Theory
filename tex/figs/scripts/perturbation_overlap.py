import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
L=1.0; x=np.linspace(-L,L,2001)
rho=np.sin(3*np.pi*(x+L)/(2*L))**2/L
fig,axs=plt.subplots(1,2,figsize=(6.5,3),sharey=True)
for ax,q in zip(axs,[3,2]):
    f=np.cos(q*np.pi*x/L); prod=rho*f
    I=trapezoid(prod,x)
    ax.plot(x,rho,'k-',label=r'$|\psi_3^{(0)}|^2$')
    ax.plot(x,f,'b--',label=r'$\cos(%d\pi x/L)$'%q)
    ax.fill_between(x,0,prod,color='orange',alpha=0.5,label='product')
    ax.axhline(0,color='gray',lw=0.5)
    ax.set_title('q = %d: integral = %.2f'%(q,abs(I) if abs(I)>1e-9 else 0.0))
    ax.set_xlabel(r'$x$ (units $\hbar=2m=1$, $L=1$)')
axs[0].set_ylabel('value (dimensionless)')
axs[0].legend(fontsize=7,loc='lower left')
plt.tight_layout()
plt.savefig("perturbation_overlap.pdf", bbox_inches="tight")
