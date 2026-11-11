import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t=qmnt.zeros(400); n=np.arange(1,401)
u=t/(2*np.pi); y=n-0.5-(u*np.log(u)-u+7/8)
w=np.linspace(0.05,4.5,6000)
P=((np.cos(np.outer(w,t))@y)**2+(np.sin(np.outer(w,t))@y)**2)/len(t)
plt.figure(figsize=(6,4)); plt.plot(w,P,lw=0.7)
for p in [2,3,5,7,11,13]:
    plt.axvline(np.log(p),ls="--",c="r",lw=0.6); plt.text(np.log(p),P.max()*1.02,str(p),ha="center",fontsize=8)
for q in [4,8,9]: plt.axvline(np.log(q),ls=":",c="g",lw=0.6)
plt.xlabel(r"angular frequency $\omega$ (per unit height)"); plt.ylabel(r"periodogram $\mathcal{P}(\omega)$ (counting units$^2$)")
plt.savefig("sparse-frequency_periodogram.pdf", bbox_inches="tight")
