import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

n = np.arange(370, 461)
t = np.asarray(qmnt.zeros(460))[369:460]
true = qmnt.fluct(t, n)
pred = qmnt.fluct(qmnt.oracle_zeros(n, qmnt.first_primes(6), mmax=3), n)

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(400.5, 460.5, color="0.9", label="beyond zero 400")
ax.plot(n, true, "o-", color="k", ms=3, lw=1, label="true fluctuation")
ax.plot(n, pred, "s-", color="tab:green", ms=3, lw=1,
        label="six-prime oracle (0 fitted parameters)")
ax.axhline(0.0, color="tab:red", ls="--", lw=1, label="null (smooth zeros)")
ax.set_xlabel("zero index n")
ax.set_ylabel("(t_n - s_n)/gap_n  (mean spacings)")
ax.set_xlim(369.5, 460.5)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Zeros 370-460: oracle vs truth")
plt.savefig("paper-arithmetic_oraclefit.pdf", bbox_inches="tight")
