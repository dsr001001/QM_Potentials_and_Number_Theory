import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L,Nb=1.0,60; n=np.arange(1,Nb+1); E0=(n*np.pi/2/L)**2
D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
C1=-0.5*((D==2)*1.0-(S==2)*1.0)
cs=np.linspace(0,40,200)
ex=[np.linalg.eigvalsh(np.diag(E0)+c*C1)[0] for c in cs]
plt.figure(figsize=(6,4))
plt.plot(cs,ex,"k",label="exact")
plt.plot(cs,E0[0]+cs/2,"--",label="first order")
plt.plot(cs,E0[0]+cs/2-cs**2/(8*np.pi**2),":",label="second order")
plt.xlabel("c (energy units)"); plt.ylabel(r"$E_1$ (energy units)"); plt.legend()
plt.savefig("perturbation_validity.pdf",bbox_inches="tight")
