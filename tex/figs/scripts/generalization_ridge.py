import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
t_tr = np.linspace(0, 2*np.pi, 10)
y_tr = np.sin(t_tr) + 0.15*rng.standard_normal(10)
t_in = rng.uniform(0, 2*np.pi, 200)
y_in = np.sin(t_in) + 0.15*rng.standard_normal(200)
t_out = np.linspace(2*np.pi, 2.5*np.pi, 50)
y_out = np.sin(t_out) + 0.15*rng.standard_normal(50)

def X(t):
    u = t/np.pi - 1.0
    return np.vander(u, 10, increasing=True)

U, S, Vt = np.linalg.svd(X(t_tr), full_matrices=False)
rms = lambda e: np.sqrt(np.mean(e**2))
mus = np.logspace(-10, 4, 150)
ein, eint, eout = [], [], []
for mu in mus:
    f = S/(S**2 + mu*S[0]**2)
    c = Vt.T @ (f*(U.T @ y_tr))
    ein.append(rms(X(t_tr) @ c - y_tr))
    eint.append(rms(X(t_in) @ c - y_in))
    eout.append(rms(X(t_out) @ c - y_out))

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(mus, ein, "-", label="in-sample")
ax.loglog(mus, eint, "--", label="out-of-sample, inside range")
ax.loglog(mus, eout, "-.", label=r"out-of-sample, $2\pi<t\leq 2.5\pi$")
ax.axhline(rms(y_in), color="C1", lw=0.6)
ax.axhline(rms(y_out), color="C2", lw=0.6)
ax.set_xlabel(r"ridge strength $\mu$ (dimensionless)")
ax.set_ylabel("rms error (dimensionless)")
ax.legend(fontsize=8)
plt.savefig("generalization_ridge.pdf", bbox_inches="tight")
