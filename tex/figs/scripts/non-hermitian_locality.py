import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
n=20
ms=np.array([m for m in range(1,61) if m!=n and (n+m)%2==1])
w=np.array([s1(n,m)**2/abs(n*n-m*m) for m in ms])
fig,ax=plt.subplots(figsize=(6,4))
ax.semilogy(ms,w,'o-',ms=4)
ax.axvline(n,color='gray',ls=':')
ax.set_xlabel("partner level m")
ax.set_ylabel(r"weight $w_{20,m}$ (energy units)")
ax.set_title(r"$A=\sin(\pi x/L)$, box with $E_n=n^2$")
plt.savefig("non-hermitian_locality.pdf", bbox_inches="tight")
