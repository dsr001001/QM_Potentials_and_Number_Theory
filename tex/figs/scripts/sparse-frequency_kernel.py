import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
J=450; dw=8/450
def K(tau):
    tau=np.asarray(tau,float)
    return np.cos(np.outer(tau,dw*np.arange(1,J+1))).sum(axis=1)
fig,ax=plt.subplots(1,2,figsize=(6.5,3))
t1=np.linspace(-3,3,2001); ax[0].plot(t1,K(t1),lw=0.8)
ax[0].set_xlabel(r"$\tau$ (height units)"); ax[0].set_ylabel(r"$K(\tau)$"); ax[0].set_title("(a) near 0")
t2=np.linspace(-50,420,8001); ax[1].plot(t2,K(t2),lw=0.5)
ax[1].axvline(2*np.pi/dw,ls=":",c="k"); ax[1].set_xlabel(r"$\tau$ (height units)"); ax[1].set_title(r"(b) recurrence at $P\approx353.4$")
plt.tight_layout(); plt.savefig("sparse-frequency_kernel.pdf", bbox_inches="tight")
