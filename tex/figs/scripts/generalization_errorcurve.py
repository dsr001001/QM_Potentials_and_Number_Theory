import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
t_tr = np.linspace(0, 2*np.pi, 15)
y_tr = np.sin(t_tr) + 0.15*rng.standard_normal(15)
t_in = rng.uniform(0, 2*np.pi, 200)
y_in = np.sin(t_in) + 0.15*rng.standard_normal(200)
t_out = np.linspace(2*np.pi, 2.4*np.pi, 50)
y_out = np.sin(t_out) + 0.15*rng.standard_normal(50)
rms = lambda e: np.sqrt(np.mean(e**2))

degs = np.arange(15)
ein, eint, eout = [], [], []
for deg in degs:
    p = np.polynomial.Polynomial.fit(t_tr, y_tr, deg)
    ein.append(max(rms(p(t_tr) - y_tr), 1e-12))
    eint.append(rms(p(t_in) - y_in))
    eout.append(rms(p(t_out) - y_out))

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(degs, ein, "o-", label="in-sample")
ax.semilogy(degs, eint, "s-", label="out-of-sample, inside range")
ax.semilogy(degs, eout, "^-", label=r"out-of-sample, $2\pi<t\leq 2.4\pi$")
ax.axhline(0.15, color="k", ls=":", lw=1, label="noise level")
ax.set_xlabel("polynomial degree (dimensionless)")
ax.set_ylabel("rms error (dimensionless)")
ax.set_ylim(1e-13, 1e4)
ax.legend(fontsize=8)
plt.savefig("generalization_errorcurve.pdf", bbox_inches="tight")
