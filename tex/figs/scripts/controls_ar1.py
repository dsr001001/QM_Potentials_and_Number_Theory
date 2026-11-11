import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def ar1(phi, n, rng):
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - phi**2)
    for j in range(1, n):
        x[j] = phi * x[j - 1] + rng.normal()
    return x

def cbb_se(x, ell, rng, B=300):
    n = len(x); nb = -(-n // ell)
    st = rng.integers(0, n, size=(B, nb))
    idx = ((st[:, :, None] + np.arange(ell)) % n).reshape(B, -1)[:, :n]
    return x[idx].mean(axis=1).std()

rng = np.random.default_rng(3)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
for k, phi in enumerate([0.5, 0.0, -0.5]):
    a1.plot(ar1(phi, 100, rng) + 8 * (1 - k), lw=0.8, label=f"phi={phi}")
a1.set_xlabel("index j"); a1.set_ylabel("X_j (offset, arb. units)")
a1.legend(fontsize=7)

phi, n = 0.5, 400
g0 = 1 / (1 - phi**2)
k = np.arange(1, n)
fn = 1 + 2 * np.sum((1 - k / n) * phi**k)
true_se = np.sqrt(g0 * fn / n)
ells = [1, 2, 3, 5, 8, 12, 20, 30, 40]
sim = []
for ell in ells:
    sim.append(np.mean([cbb_se(ar1(phi, n, rng), ell, rng) for _ in range(40)]) / true_se)
exact = []
for ell in ells:
    kk = np.arange(1, ell)
    f = 1 + 2 * np.sum((1 - kk / ell) * phi**kk)
    exact.append(np.sqrt(f / fn))
a2.plot(ells, sim, "o-", label="simulated")
a2.plot(ells, exact, "k--", label="Eq. (cbbvar), true autocov.")
a2.axhline(1, color="gray", lw=0.8)
a2.set_xlabel("block length l"); a2.set_ylabel("bootstrap SE / true SE")
a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("controls_ar1.pdf", bbox_inches="tight")
