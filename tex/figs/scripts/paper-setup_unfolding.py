import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import lambertw
import qmnt

def Nsm(t):
    u = t / (2*np.pi)
    return u*np.log(u) - u + 7/8

def smooth_zero(n):
    z = (n - 11/8) / np.e
    return 2*np.pi*(n - 11/8) / np.real(lambertw(z, 0))

t = np.asarray(qmnt.zeros(500), float)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
tt = np.linspace(10, 100, 3000)
N = np.searchsorted(t, tt, side="right")
a1.plot(tt, N, drawstyle="steps-post", lw=1, label="$N(t)$")
a1.plot(tt, Nsm(tt), "r--", lw=1, label="$N_{\\mathrm{sm}}(t)$")
a1.set_xlabel("height $t$ (dimensionless)")
a1.set_ylabel("number of zeros")
a1.legend(fontsize=8); a1.set_title("(a)", fontsize=9)

n = np.arange(1, 501)
s = smooth_zero(n)
g = 2*np.pi / np.log(s / (2*np.pi))
f = (t - s) / g
a2.axvspan(400.5, 500.5, color="orange", alpha=0.25)
a2.plot(n, f, ".", ms=2)
a2.axhline(0, color="k", lw=0.5)
a2.set_xlabel("zero index $n$")
a2.set_ylabel("$f_n$ (mean spacings)")
a2.set_title("(b)", fontsize=9)
plt.tight_layout()
plt.savefig("paper-setup_unfolding.pdf", bbox_inches="tight")
