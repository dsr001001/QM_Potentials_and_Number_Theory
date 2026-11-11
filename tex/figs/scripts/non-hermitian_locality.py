import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
n=20; ms=np.array([m for m in range(1,61) if m!=n])
w=np.array([s1(n,m)**2/abs(n**2-m**2) for m in ms])
fig,ax=plt.subplots(figsize=(6,4))
ok=w>0
ax.semilogy(ms[ok],w[ok],"o")
ax.set_xlabel("partner level m"); ax.set_ylabel(r"weight $w_{20,m}$ (energy units)")
plt.savefig("non-hermitian_locality.pdf",bbox_inches="tight")
