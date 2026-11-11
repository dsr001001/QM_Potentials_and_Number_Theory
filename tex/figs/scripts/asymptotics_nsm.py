import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mpmath
t=np.linspace(0.05,60,800); u=t/(2*np.pi); N=u*np.log(u)-u+7/8
z=[float(mpmath.zetazero(k).imag) for k in range(1,14)]
plt.figure(figsize=(6,4)); plt.plot(t,N,'k',label="N_sm(t)")
cnt=[sum(1 for zz in z if zz<=tt) for tt in t]; plt.step(t,cnt,'b',where='post',label="N(t)")
for n in range(1,9): plt.axhline(n-0.5,ls='--',lw=0.5,color='gray')
plt.axvline(2*np.pi,ls=':',color='r')
plt.xlabel("height t (dimensionless)"); plt.ylabel("number of zeros"); plt.legend()
plt.savefig("asymptotics_nsm.pdf", bbox_inches="tight")
