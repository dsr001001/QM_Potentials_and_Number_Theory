import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
z = qmnt.zeros(40); z = z[z < 100]
t = np.linspace(10, 100, 2000)
Nst = np.searchsorted(z, t, side="right")
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3))
a.plot(t, Nst, drawstyle="steps-post", label="N(t)")
a.plot(t, qmnt.Nsm(t), label="Nsm(t)")
a.set_xlabel("t"); a.set_ylabel("count"); a.legend(); a.set_title("(a)")
uu = qmnt.Nsm(t)
b.plot(uu, Nst, drawstyle="steps-post", label="N")
b.plot(uu, uu + 0.5, label="u + 1/2")
b.set_xlabel("u = Nsm(t)"); b.set_ylabel("count"); b.legend(); b.set_title("(b)")
plt.tight_layout()
plt.savefig("level-statistics_staircase.pdf", bbox_inches="tight")
