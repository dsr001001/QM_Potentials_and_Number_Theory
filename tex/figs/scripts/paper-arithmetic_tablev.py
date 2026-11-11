import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

mid = np.array([450.5, 750.5, 1500.5, 3500.5, 7500.5])
null = np.array([0.263, 0.270, 0.279, 0.288, 0.296])
data = {"{2}": [0.198, 0.208, 0.219, 0.232, 0.242],
        "3 primes": [0.116, 0.127, 0.139, 0.156, 0.169],
        "6 primes": [0.074, 0.083, 0.091, 0.107, 0.119],
        "25 primes": [0.043, 0.050, 0.054, 0.060, 0.067]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(mid, null, "k--", label="null")
for k, v in data.items():
    a1.plot(mid, v, "o-", ms=3, label=k)
    a2.plot(mid, np.array(v) / null, "o-", ms=3, label=k)
for a in (a1, a2):
    a.set_xscale("log"); a.set_xlabel("holdout midpoint (zero index)")
a1.set_ylabel("held-out rms (mean spacings)")
a2.set_ylabel("rms / null (dimensionless)")
a1.legend(fontsize=7); a2.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("paper-arithmetic_tablev.pdf", bbox_inches="tight")
