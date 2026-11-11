import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb=1.0,60; n=np.arange(1,Nb+1); E0=(n*np.pi/2/L)**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
C1=-0.5*((D==2)*1.0-(S==2)*1.0)
def g(c):
    w,v=np.linalg.eigh(np.diag(E0)+c*C1); return w[0],v[:,0]@C1@v[:,0]
cs=np.linspace(0,1.5,200); plt.figure(figsize=(6,4))
plt.plot(cs,[g(c)[0] for c in cs],"k",label=r"$E_1(c)$"); plt.axhline(3,color="gray")
E,A=g(0); plt.plot(cs,E+A*cs,"--",label="tangent at 0")
c1=(3-E)/A; E,A=g(c1); plt.plot(cs,E+A*(cs-c1),":",label=r"tangent at $c_1$")
plt.xlabel("c (energy units)"); plt.ylabel(r"$E_1$ (energy units)"); plt.legend()
plt.savefig("perturbation_newton.pdf",bbox_inches="tight")
