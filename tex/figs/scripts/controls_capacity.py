import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import qmnt
n = np.arange(101, 201)
s = qmnt.smooth_zeros(n)
y = -np.asarray(qmnt.Nfl(s, np.array([2]), mmax=3))
rng = np.random.default_rng(6)
def phase_randomize(d):
    D = np.fft.rfft(d)
    ph = rng.uniform(0, 2*np.pi, len(D)); ph[0] = 0.0
    if len(d) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(D*np.exp(1j*ph), n=len(d))
j = np.arange(100)
C = np.cos(np.pi*np.outer(j + 0.5, np.arange(100))/100)
def rms_res(X, v):
    c = np.linalg.lstsq(X, v, rcond=None)[0]
    return np.sqrt(np.mean((v - X @ c)**2))
Ms = list(range(1, 101, 3))
ph = [phase_randomize(y) for _ in range(20)]
pm = [rng.permutation(y) for _ in range(20)]
real = [rms_res(C[:, :M], y) for M in Ms]
phc = [np.mean([rms_res(C[:, :M], v) for v in ph]) for M in Ms]
pmc = [np.mean([rms_res(C[:, :M], v) for v in pm]) for M in Ms]
S = np.column_stack([f(m*s*np.log(2)) for m in (1, 2, 3) for f in (np.sin, np.cos)])
ratio = lambda v: rms_res(S, v)/np.sqrt(np.mean(v**2))
rph = [ratio(v) for v in ph]; rpm = [ratio(v) for v in pm]
rr = [ratio(y), np.mean(rph), np.mean(rpm)]
re_ = [0.0, np.std(rph), np.std(rpm)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(Ms, real, 'k-', label='real target')
a1.plot(Ms, phc, '--', color='tab:orange', label='phase-randomized (avg)')
a1.plot(Ms, pmc, ':', color='tab:blue', label='permuted (avg)')
a1.set_xlabel('basis size M')
a1.set_ylabel('residual rms (mean spacings)')
a1.legend(fontsize=7)
a2.bar([0, 1, 2], rr, yerr=re_, color=['k', 'tab:orange', 'tab:blue'])
a2.set_xticks([0, 1, 2])
a2.set_xticklabels(['real', 'phase-rand.', 'permuted'], fontsize=8)
a2.set_ylabel('residual rms / target rms')
fig.tight_layout()
plt.savefig('controls_capacity.pdf', bbox_inches='tight')
