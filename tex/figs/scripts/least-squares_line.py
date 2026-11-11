import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t=np.array([0,1,2,3.]); y=np.array([1,3,4,4.])
c1,c0=np.polyfit(t,y,1)
fig,ax=plt.subplots(figsize=(6,4))
tt=np.linspace(-0.3,3.3,100); ax.plot(tt,c0+c1*tt,'b-',label=f"y = {c0:.1f} + {c1:.1f} t")
for ti,yi in zip(t,y): ax.plot([ti,ti],[yi,c0+c1*ti],'r-')
ax.plot(t,y,'ko',label="data"); ax.plot([t.mean()],[y.mean()],'o',mfc='w',mec='k',ms=9,label="point of averages")
ax.set_xlabel("time t (s)"); ax.set_ylabel("position y (m)"); ax.legend()
plt.savefig("least-squares_line.pdf", bbox_inches="tight")
