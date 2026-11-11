import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def autocorr(x, kmax):
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x**2)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / den for k in range(kmax + 1)])

n = np.arange(401, 501)
y = qmnt.Nfl(qmnt.smooth_zeros(n), [2], mmax=3)
z = np.random.default_rng(1).normal(size=100)
lags = np.arange(21)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lags, autocorr(y, 20), "bo-", label="RE2 at smooth zeros 401-500")
ax.plot(lags, autocorr(z, 20), "rs--", label="independent Gaussian noise")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag $k$ (levels)")
ax.set_ylabel("autocorrelation $r_k$ (dimensionless)")
ax.legend()
plt.savefig("probability_autocorr.pdf", bbox_inches="tight")
