import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lam=np.linspace(-2,2,400); v=0.5
plt.figure(figsize=(6,4))
plt.plot(lam,lam,"k--",lam,-lam,"k--")
plt.plot(lam,np.sqrt(lam**2+v**2),"b",lam,-np.sqrt(lam**2+v**2),"b")
plt.xlabel(r"$\lambda$ (energy units)"); plt.ylabel("eigenvalue (energy units)")
plt.savefig("perturbation_repulsion.pdf",bbox_inches="tight")
