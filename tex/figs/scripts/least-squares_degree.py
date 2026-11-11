import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t=0.7*np.arange(10); eps=np.array([.15,-.12,.10,-.15,.12,-.10,.14,-.13,.11,-.14]); y=np.sin(t)+eps
u=lambda tt: tt/3.15-1; tt=np.linspace(0,6.3,2000)
tr=[];te=[]
for d in range(10):
    c=np.linalg.lstsq(np.vander(u(t),d+1),y,rcond=None)[0]
    tr.append(max(np.sqrt(np.mean((np.vander(u(t),d+1)@c-y)**2)),1e-16))
    te.append(np.sqrt(np.mean((np.vander(u(tt),d+1)@c-np.sin(tt))**2)))
fig,ax=plt.subplots(figsize=(6,4))
ax.semilogy(range(10),tr,'o-',label="training error"); ax.semilogy(range(10),te,'s-',label="test error")
ax.set_xlabel("polynomial degree d"); ax.set_ylabel("rms error (dimensionless)"); ax.legend()
plt.savefig("least-squares_degree.pdf", bbox_inches="tight")
