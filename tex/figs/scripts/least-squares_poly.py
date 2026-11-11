import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t=0.7*np.arange(10); eps=np.array([.15,-.12,.10,-.15,.12,-.10,.14,-.13,.11,-.14]); y=np.sin(t)+eps
u=lambda tt: tt/3.15-1
tt=np.linspace(0,6.3,600)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(tt,np.sin(tt),color='0.6',lw=3,label="sin t")
for d in [1,3,9]:
    c=np.linalg.lstsq(np.vander(u(t),d+1),y,rcond=None)[0]
    ax.plot(tt,np.vander(u(tt),d+1)@c,label=f"degree {d}")
ax.plot(t,y,'ko',label="data"); ax.set_ylim(-2,2)
ax.set_xlabel("t (dimensionless)"); ax.set_ylabel("y (dimensionless)"); ax.legend(fontsize=8)
plt.savefig("least-squares_poly.pdf", bbox_inches="tight")
