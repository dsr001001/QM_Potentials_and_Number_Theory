import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def Nsm(t): u=t/(2*np.pi); return u*np.log(u)-u+7/8
t=np.linspace(740,752,3000); L2=np.log(2)
F1=-sum(np.sin(m*t*L2)/(m*2**(m/2)) for m in (1,2,3))/np.pi; F2=0.6*np.sin(3*t)
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
for a,F,ti in [(ax[0],F1,"(a) RE2"),(ax[1],F2,r"(b) $0.6\sin 3t$")]:
    a.plot(t,Nsm(t)+F); a.plot(t,Nsm(t),":k")
    for k in range(int(Nsm(t[0]))-1,int(Nsm(t[-1]))+3): a.axhline(k-0.5,c="gray",lw=0.5)
    a.set_title(ti); a.set_xlabel("height t (dimensionless)")
ax[0].set_ylabel(r"$N_{\rm sm}+F$ (count)")
plt.tight_layout(); plt.savefig("sparse-frequency_roots.pdf", bbox_inches="tight")
