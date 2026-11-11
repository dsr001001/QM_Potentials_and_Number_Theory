import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
hold = [(401, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)]
null_p = [0.263, 0.270, 0.279, 0.288, 0.296]      # paper Table V
six_p = [0.074, 0.083, 0.091, 0.107, 0.119]
tw5_p = [0.043, 0.050, 0.054, 0.060, 0.067]
mid = [np.sqrt(a * b) for a, b in hold]
t = qmnt.zeros(2000)
P6 = qmnt.first_primes(6)
mn, m6 = [], []
for a, b in hold[:3]:
    n = np.arange(a, b + 1)
    mn.append(qmnt.rms(qmnt.fluct(t[n - 1], n)))
    m6.append(qmnt.score(qmnt.oracle_zeros(n, P6), n)[0])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mid, null_p, "k--o", mfc="none", label="null (Table V)")
ax.plot(mid, six_p, "b-o", mfc="none", label="6 primes (Table V)")
ax.plot(mid, tw5_p, "g-o", mfc="none", label="25 primes (Table V)")
ax.plot(mid[:3], mn, "ks", label="null (qmnt)")
ax.plot(mid[:3], m6, "bs", label="6 primes (qmnt)")
ax.set_xscale("log")
ax.set_xlabel("zero index n (geometric mean of holdout)")
ax.set_ylabel("held-out rms (mean spacings)")
ax.set_ylim(0, 0.32)
ax.legend(fontsize=8)
plt.savefig("paper-critique_height.pdf", bbox_inches="tight")
