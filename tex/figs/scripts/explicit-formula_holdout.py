import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
n = np.arange(401, 501)
z = np.asarray(qmnt.zeros(500))
true = qmnt.fluct(z[n-1], n)
pred = qmnt.fluct(np.asarray(qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3)), n)
ks = np.arange(1, 26)
rm = [qmnt.score(qmnt.oracle_zeros(n, qmnt.first_primes(k), mmax=3), n)[0] for k in ks]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3))
a.plot(n, true, 'k.', ms=3, label="true"); a.plot(n, pred, lw=1, label="6 primes")
a.set_xlabel("zero index n"); a.set_ylabel("fluctuation (mean spacings)"); a.legend(fontsize=7)
b.plot(ks, rm, 'o-', ms=3); b.axhline(qmnt.rms(true), ls="--", color="gray", label="null")
b.set_xlabel("number of primes k"); b.set_ylabel("rms error (mean spacings)"); b.legend(fontsize=7)
plt.tight_layout()
plt.savefig("explicit-formula_holdout.pdf", bbox_inches="tight")
