import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
V=np.linspace(qmnt.V0,1000,800); x=np.array([qmnt.wu_sprung_x(v) for v in V])
t=qmnt.zeros(300); x1,x2=qmnt.wu_sprung_x(t[200]),qmnt.wu_sprung_x(t[299])
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(x,V,'k')
ax.axvspan(x1,x2,color='0.85'); ax.axhline(t[200],ls=':'); ax.axhline(t[299],ls=':')
ax.set_xlabel('x (units $\\hbar=2m=1$)'); ax.set_ylabel('V'); ax.set_xlim(0,45)
plt.savefig("wkb_turnmap.pdf", bbox_inches="tight")
