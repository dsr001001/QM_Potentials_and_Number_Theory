import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt
t=qmnt.zeros(400); n=np.arange(1,401); u=t/(2*np.pi); y=n-0.5-(u*np.log(u)-u+7/8)
tr,va=slice(0,300),slice(300,400); tt,yy=t[tr],y[tr]
def per(r,w): return (np.cos(np.outer(w,tt))@r)**2+(np.sin(np.outer(w,tt))@r)**2
def des(x,om): return np.hstack([np.sin(np.outer(x,om)),np.cos(np.outer(x,om))])
T=tt[-1]-tt[0]; st=0.25*2*np.pi/T; coarse=np.arange(0.05,8,st)
om=[]; r=yy.copy(); trn=[]; val=[]; Ks=range(1,41)
for k in Ks:
    w0=coarse[np.argmax(per(r,coarse))]; fine=np.linspace(w0-st,w0+st,201); om.append(fine[np.argmax(per(r,fine))])
    X=des(tt,np.array(om)); c=np.linalg.lstsq(X,yy,rcond=None)[0]; r=yy-X@c
    trn.append(np.sqrt(np.mean(r**2))); val.append(np.sqrt(np.mean((y[va]-des(t[va],np.array(om))@c)**2)))
plt.figure(figsize=(6,4)); plt.plot(Ks,trn,"o-",ms=3,label="training (1-300)"); plt.plot(Ks,val,"o-",mfc="none",ms=3,label="validation (301-400)")
plt.axhline(np.sqrt(np.mean(yy**2)),ls="--",c="C0"); plt.axhline(np.sqrt(np.mean(y[va]**2)),ls="--",c="C1")
plt.xlabel("number of frequencies K"); plt.ylabel("rms of y - F (counting units)"); plt.legend()
plt.savefig("sparse-frequency_greedy.pdf", bbox_inches="tight")
