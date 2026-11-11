import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x=np.logspace(-5,0,400)
fig,(a,b)=plt.subplots(1,2,figsize=(6.5,3))
for lam in [1e-2,1e-4,1e-6]:
    a.semilogx(x,x**2/(x**2+lam),label=f"ridge lambda={lam:g}")
    b.loglog(x,x/(x**2+lam),label=f"lambda={lam:g}")
a.semilogx(x,(x>=1e-2).astype(float),'k--',label="TSVD")
b.loglog(x,1/x,'k--',label="1/sigma")
a.set_xlabel("sigma/sigma_1"); a.set_ylabel("filter factor f"); a.legend(fontsize=6)
b.set_xlabel("sigma/sigma_1"); b.set_ylabel("gain x sigma_1"); b.legend(fontsize=6)
plt.tight_layout(); plt.savefig("least-squares_filter.pdf", bbox_inches="tight")
