import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A = np.array([[2.0, 1.0], [1.0, 2.0]])
th = np.linspace(0, 2*np.pi, 400)
circ = np.vstack([np.cos(th), np.sin(th)])
ell = A @ circ

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(circ[0], circ[1], "k--", lw=1, label="unit circle")
ax.plot(ell[0], ell[1], "b-", lw=1.5, label="image under $A$")

u1 = np.array([1, 1])/np.sqrt(2)
u2 = np.array([1, -1])/np.sqrt(2)
v = np.array([1.0, 0.0])
def arrow(p, color, lw=2, ls="-"):
    ax.annotate("", xy=p, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))
arrow(u1, "red", 1.5); arrow(3*u1, "red", 2.5)
arrow(u2, "green", 1.5); arrow(1*u2*1.0, "green", 2.5)
arrow(v, "gray", 1.5); arrow(A @ v, "black", 1.5)
ax.text(3*u1[0]+0.05, 3*u1[1], r"$Au_1=3u_1$", color="red")
ax.text(u2[0]+0.05, u2[1]-0.2, r"$Au_2=u_2$", color="green")
ax.text(1.02, -0.2, r"$(1,0)$", color="gray")
ax.text(2.02, 1.0, r"$A(1,0)=(2,1)$", color="black")
ax.set_aspect("equal")
ax.set_xlim(-3.2, 3.8); ax.set_ylim(-2.6, 2.6)
ax.set_xlabel("first component (dimensionless)")
ax.set_ylabel("second component (dimensionless)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("eigen_stretch.pdf", bbox_inches="tight")
