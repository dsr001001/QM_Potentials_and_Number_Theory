import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

def phase_randomize(d, rng):
    D = np.fft.rfft(d)
    ph = rng.uniform(0, 2*np.pi, len(D)); ph[0] = 0.0
    if len(d) % 2 == 0: ph[-1] = 0.0
    return np.fft.irfft(D * np.exp(1j*ph), n=len(d))

def resid_rms(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.sqrt(np.mean((y - X @ c)**2))

n = np.arange(101, 201)
s = qmnt.smooth_zeros(n)
d = -qmnt.Nfl(s, [2])
j = np.arange(100)
rng = np.random.default_rng(5)
Ms = np.arange(1, 101, 3).tolist() + [100]
phs = [phase_randomize(d, rng) for _ in range(20)]
pms = [rng.permutation(d) for _ in range(20)]
def curve(y):
    return [resid_rms(np.cos(np.pi*np.outer(j + 0.5, np.arange(M))/100), y) for M in Ms]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(Ms, curve(d), "k-", label="real target")
a1.plot(Ms, np.mean([curve(y) for y in phs], 0), "C1--", label="phase-rand. (mean)")
a1.plot(Ms, np.mean([curve(y) for y in pms], 0), "C2:", label="permuted (mean)")
a1.set_xlabel("generic basis size M"); a1.set_ylabel("residual rms (mean spacings)")
a1.legend(fontsize=7)
X = np.column_stack([f(m*s*np.log(2)) for m in (1, 2, 3) for f in (np.sin, np.cos)])
rel = lambda y: resid_rms(X, y) / np.sqrt(np.mean(y**2))
a2.bar([0], [rel(d)], color="k")
a2.boxplot([[rel(y) for y in phs], [rel(y) for y in pms]], positions=[1, 2])
a2.set_xticks([0, 1, 2]); a2.set_xticklabels(["real", "phase-rand.", "permuted"], fontsize=8)
a2.set_ylabel("structured-model residual / target rms")
plt.tight_layout()
plt.savefig("controls_capacity.pdf", bbox_inches="tight")
