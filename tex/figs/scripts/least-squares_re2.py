import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L2=np.log(2)
F=lambda t: -sum(np.sin(m*t*L2)/(m*2**(m/2)) for m in (1,2,3))/np.pi
def des(t,w): return np.hstack([np.sin(np.outer(t,w)),np.cos(np.outer(t,w))])
ts=np.linspace(10,60,80); tt=np.linspace(0,100,2000)
w3=L2*np.array([1,2,3.]); wg=8*np.arange(1,61)/60
c3=np.linalg.lstsq(des(ts,w3),F(ts),rcond=None)[0]; cg=np.linalg.lstsq(des(ts,wg),F(ts),rcond=None)[0]
fig,ax=plt.subplots(figsize=(6,4)); ax.axvspan(10,60,color='y',alpha=0.2)
ax.plot(tt,F(tt),color='0.6',lw=3,label="true wave")
ax.plot(tt,des(tt,w3)@c3,'b--',label="3 correct frequencies")
ax.plot(tt,des(tt,wg)@cg,'r-',lw=1,label="120-coef uniform grid")
ax.plot(ts,F(ts),'k.',ms=3); ax.set_ylim(-1,1)
ax.set_xlabel("t (dimensionless)"); ax.set_ylabel("F (dimensionless)"); ax.legend(fontsize=8)
plt.savefig("least-squares_re2.pdf", bbox_inches="tight")
