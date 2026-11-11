import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(2)
t_tr = np.linspace(0, 2*np.pi, 10)
y_tr = np.sin(t_tr) + 0.15*rng.standard_normal(10)
t_in = rng.uniform(0, 2*np.pi, 200); y_in = np.sin(t_in) + 0.15*rng.standard_normal(200)
t_out = np.linspace(2*np.pi, 2.5*np.pi, 50); y_out = np.sin(t_out) + 0.15*rng.standard_normal(50)
u = lambda t: t/np.pi - 1.0
X = lambda t: np.vander(u(t), 10, increasing=True)
U, S, Vt = np.linalg.svd(X(t_tr), full_matrices=False)
rms = lambda e: np.sqrt(np.mean(e**2))
mus = np.logspace(-12, 4, 161); E1 = []; E2 = []; E3 = []
for mu in mus:
    c = Vt.T @ ((S/(S**2 + mu*S[0]**2)) * (U.T @ y_tr))
    E1.append(rms(X(t_tr)@c - y_tr)); E2.append(rms(X(t_in)@c - y_in)); E3.append(rms(X(t_out)@c - y_out))
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(mus, E1, "-", label="in-sample")
ax.loglog(mus, E2, "--", label="out-of-sample, in range")
ax.loglog(mus, E3, "-.", label=r"out-of-sample, $2\pi<t\leq2.5\pi$")
for yv in [rms(y_tr), rms(y_in), rms(y_out)]:
    ax.axhline(yv, color="0.6", lw=0.6)
ax.set_xlabel(r"ridge strength $\mu$ (dimensionless)"); ax.set_ylabel("rms error (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("generalization_ridge.pdf", bbox_inches="tight")
