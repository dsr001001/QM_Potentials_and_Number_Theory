import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt

L = qmnt.L_of(1000)
xs = np.linspace(-L, L, 801)
Vs = np.array([qmnt.wu_sprung_V(float(xi)) for xi in xs])
E500 = 811.1843588
xt = qmnt.wu_sprung_x(E500)

N = 3000
x = qmnt.grid(L, N)
h = 2*L/(N + 1)
V = np.array([qmnt.wu_sprung_V(float(xi)) for xi in x])
E, vec = qmnt.fd_levels(V, L, k=4, return_vectors=True)
vec = np.asarray(vec)
if vec.shape[0] != N:
    vec = vec.T

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3))
a1.plot(xs, Vs, "k", lw=1.2)
a1.axvline(-L, color="k", lw=3); a1.axvline(L, color="k", lw=3)
a1.axhline(E500, color="C3", ls="--", lw=1)
a1.plot([-xt, xt], [E500, E500], "o", color="C3", ms=4)
a1.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
a1.set_ylabel(r"$V(x)$ (units $\hbar=2m=1$)")
a1.set_title("Wu--Sprung potential, scale 1", fontsize=9)
m = np.abs(x) < 12
a2.plot(x[m], V[m], "k", lw=1.2)
for j in range(4):
    psi = vec[:, j]/np.sqrt(np.sum(vec[:, j]**2)*h)
    if psi[np.argmax(np.abs(psi))] < 0:
        psi = -psi
    a2.axhline(E[j], color="0.85", lw=0.8)
    a2.plot(x[m], E[j] + 6*psi[m], lw=1.3)
a2.set_ylim(5, 45)
a2.set_xlabel(r"$x$ (units $\hbar=2m=1$)")
a2.set_ylabel(r"energy; $E_n+6\,\psi_n$")
a2.set_title("lowest four states", fontsize=9)
plt.tight_layout()
plt.savefig("bound-states_wusprung.pdf", bbox_inches="tight")
