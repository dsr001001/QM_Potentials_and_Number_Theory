import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Paper Table II, quoted exactly
lab = ["cos 400","Cheb 200","Cheb 400","Cheb 600","Cheb ridge","hat 200","hat 400","hat 600"]
rms = [0.369,0.368,0.291,0.733,0.264,0.342,0.340,0.330]
rlo = [0.32,0.32,0.25,0.64,0.24,0.30,0.29,0.29]
rhi = [0.42,0.42,0.33,0.82,0.29,0.38,0.39,0.37]
cor = [-0.07,-0.05,0.18,0.01,0.03,-0.05,-0.19,0.09]
clo = [-0.22,-0.25,-0.02,-0.20,-0.21,-0.24,-0.37,-0.13]
chi = [0.07,0.14,0.37,0.23,0.28,0.14,-0.01,0.30]
y = np.arange(len(lab))[::-1]
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
ax[0].errorbar(rms, y, xerr=[np.array(rms)-rlo, np.array(rhi)-rms], fmt="o", capsize=3)
ax[0].axvline(0.263, ls="--", color="k", lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=8)
ax[0].set_xlabel("OOS rms (mean spacings)")
ax[1].errorbar(cor, y, xerr=[np.array(cor)-clo, np.array(chi)-cor], fmt="s", color="C3", capsize=3)
ax[1].axvline(0.0, ls="--", color="k", lw=1)
ax[1].set_xlabel("OOS correlation")
plt.tight_layout()
plt.savefig("paper-memorization_bases.pdf", bbox_inches="tight")
