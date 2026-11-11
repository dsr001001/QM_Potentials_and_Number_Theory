import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
ranges = [(401, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)]
null_p = [0.263, 0.270, 0.279, 0.288, 0.296]
six_p = [0.074, 0.083, 0.091, 0.107, 0.119]
tw5_p = [0.043, 0.050, 0.054, 0.060, 0.067]
xg = [np.sqrt(a * b) for a, b in ranges]
z = qmnt.zeros(2000)
P6 = qmnt.first_primes(6)
null_q, six_q = [], []
for a, b in ranges[:3]:
    n = np.arange(a, b + 1)
    t = z[n - 1]
    null_q.append(qmnt.rms(qmnt.fluct(t, n)))
    r, c = qmnt.score(qmnt.oracle_zeros(n, P6), n)
    six_q.append(r)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xg, null_p, "o-", mfc="none", color="k", label="null (Table V)")
ax.plot(xg, six_p, "s-", mfc="none", color="tab:blue", label="6 primes (Table V)")
ax.plot(xg, tw5_p, "^-", mfc="none", color="tab:green", label="25 primes (Table V)")
ax.plot(xg[:3], null_q, "o", color="k", label="null (qmnt)")
ax.plot(xg[:3], six_q, "s", color="tab:blue", label="6 primes (qmnt)")
ax.set_xscale("log")
ax.set_xlabel("zero index n (geometric mean of holdout range)")
ax.set_ylabel("held-out rms (mean spacings)")
ax.legend(fontsize=7)
plt.savefig("paper-critique_height.pdf", bbox_inches="tight")
