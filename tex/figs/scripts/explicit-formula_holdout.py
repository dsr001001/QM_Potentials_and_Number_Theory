import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

n = np.arange(401, 501)
z = qmnt.zeros(500)
true = qmnt.fluct(z[n - 1], n)
pred = qmnt.fluct(qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3), n)
null = qmnt.rms(true)
ks = np.arange(1, 31)
rmss = []
for k in ks:
    r, c = qmnt.score(qmnt.oracle_zeros(n, qmnt.first_primes(int(k)), mmax=3), n)
    rmss.append(r)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(n, true, "k.", ms=4, label="true")
a1.plot(n, pred, "-", lw=1.3, label="6 primes")
a1.set_xlabel("zero index n"); a1.set_ylabel("fluctuation (mean spacings)")
a1.legend(fontsize=7)
a2.plot(ks, rmss, "o-", ms=3)
a2.axhline(null, ls="--", color="gray", label="null")
a2.set_xlabel("number of primes k"); a2.set_ylabel("rms error (mean spacings)")
a2.set_ylim(0, 0.3); a2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("explicit-formula_holdout.pdf", bbox_inches="tight")
