import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
def Nsm(t):
    u = np.asarray(t)/(2*np.pi); return u*np.log(u) - u + 7/8
s450 = brentq(lambda x: Nsm(x) - 449.5, 100, 5000)
tt = np.linspace(s450 - 8, s450 + 8, 4000)
F1 = -sum(np.sin(m*tt*np.log(2))/(m*2**(m/2)) for m in (1, 2, 3))/np.pi
F2 = 0.6*np.sin(3*tt)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for a_, F, title in ((ax[0], F1, "(a) RE2"), (ax[1], F2, r"(b) $0.6\sin 3t$")):
    a_.plot(tt, Nsm(tt) + F, color="k", lw=1)
    a_.plot(tt, Nsm(tt), ":", color="gray")
    lo, hi = Nsm(tt[0]), Nsm(tt[-1])
    for nn in range(int(np.ceil(lo + 0.5)), int(np.floor(hi + 0.5)) + 1):
        a_.axhline(nn - 0.5, color="tab:blue", lw=0.4)
    a_.set_xlabel(r"height $t$ (dimensionless)")
    a_.set_title(title, fontsize=9)
ax[0].set_ylabel(r"$N_{\rm sm}(t)+F(t)$ (counting units)")
plt.tight_layout()
plt.savefig("sparse-frequency_roots.pdf", bbox_inches="tight")
