import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
for a,x0,n in zip(ax,[1.5,1.0],[3,4]):
    x=np.linspace(-6,6,400); a.plot(x,np.arctan(x),'k'); a.axhline(0,color='gray',lw=0.5)
    xk=x0
    for k in range(n):
        xn=xk-(1+xk**2)*np.arctan(xk)
        a.plot([xk,xk],[0,np.arctan(xk)],'b:'); a.plot([xk,xn],[np.arctan(xk),0],'r--')
        xk=xn
    a.set_xlim(-6,6); a.set_xlabel("x (dimensionless)"); a.set_ylabel("arctan x"); a.set_title(f"x0 = {x0}")
plt.tight_layout(); plt.savefig("asymptotics_newtonfail.pdf", bbox_inches="tight")
