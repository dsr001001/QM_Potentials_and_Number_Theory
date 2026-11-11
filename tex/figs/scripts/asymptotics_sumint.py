import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f=lambda x:1/x; x=np.linspace(1,7,400)
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
for a,off in zip(ax,[0,-1]):
    a.plot(x,f(x),'k')
    ks=range(1,7) if off==0 else range(2,8)
    for k in ks:
        if k+off+1>7.01: continue
        a.bar(k+off,f(k),width=1,align='edge',alpha=0.3,edgecolor='b')
    a.set_xlabel("x (dimensionless)"); a.set_ylabel("f(x)=1/x")
ax[0].set_title("heights f(k) on [k,k+1]"); ax[1].set_title("heights f(k) on [k-1,k]")
plt.tight_layout(); plt.savefig("asymptotics_stirling.pdf" if False else "asymptotics_sumint.pdf", bbox_inches="tight")
