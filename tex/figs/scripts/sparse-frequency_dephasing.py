import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(6.5,3)); t=np.linspace(0,6000,2000); tc=347
for d in [3e-4,1e-3,3e-3]: ax[0].plot(t,np.cos(d*(t-tc)),label=r"$\delta\omega$=%g"%d)
ax[0].axvspan(14,680,alpha=0.2); ax[0].set_xlabel("height t (dimensionless)"); ax[0].set_ylabel(r"corr $\cos\phi$"); ax[0].legend(fontsize=7)
p=np.linspace(0,np.pi,300); ax[1].plot(p,2*np.abs(np.sin(p/2)),label="rms/null"); ax[1].plot(p,np.cos(p),label="corr")
ax[1].axvline(np.pi/3,ls=":",c="k"); ax[1].axhline(1,ls=":",c="gray"); ax[1].set_xlabel(r"phase error $\phi$ (rad)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("sparse-frequency_dephasing.pdf", bbox_inches="tight")
