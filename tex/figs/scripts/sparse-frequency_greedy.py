import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t = qmnt.zeros(400); n = np.arange(1, 401)
u = t/(2*np.pi); y = n - 0.5 - (u*np.log(u) - u + 7/8)
tr, ytr, tv, yv = t[:300], y[:300], t[300:], y[300:]
def design(tt, om):
    return np.hstack([np.sin(np.outer(tt, om)), np.cos(np.outer(tt, om))])
def pg(tt, r, w):
    return ((np.cos(np.outer(w, tt)) @ r)**2 + (np.sin(np.outer(w, tt)) @ r)**2)/len(tt)
T = tr[-1] - tr[0]; step = 0.25*2*np.pi/T
coarse = np.arange(0.05, 8.0, step)
om, r = [], ytr.copy(); trr, vr = [], []
Kmax = 40; lam = 1e-8
for k in range(Kmax):
    w0 = coarse[np.argmax(pg(tr, r, coarse))]
    fine = np.linspace(w0 - step, w0 + step, 201)
    om.append(fine[np.argmax(pg(tr, r, fine))])
    X = design(tr, np.array(om))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    c = Vt.T @ (s/(s**2 + lam*s[0]**2) * (U.T @ ytr))
    r = ytr - X @ c
    trr.append(np.sqrt(np.mean(r**2)))
    vr.append(np.sqrt(np.mean((yv - design(tv, np.array(om)) @ c)**2)))
Ks = np.arange(1, Kmax+1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ks, trr, "o-", ms=4, label="training (zeros 1--300)")
ax.plot(Ks, vr, "o-", ms=4, mfc="white", label="validation (zeros 301--400)")
ax.axhline(np.sqrt(np.mean(ytr**2)), ls="--", color="tab:blue", lw=0.8)
ax.axhline(np.sqrt(np.mean(yv**2)), ls="--", color="tab:orange", lw=0.8)
ax.set_xlabel(r"number of frequencies $K$")
ax.set_ylabel(r"rms of $y_n-F(t_n)$ (counting units)")
ax.legend(fontsize=8)
plt.savefig("sparse-frequency_greedy.pdf", bbox_inches="tight")
