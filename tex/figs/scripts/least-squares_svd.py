import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A=np.array([[3,0],[4,5.]]); U,s,Vh=np.linalg.svd(A)
th=np.linspace(0,2*np.pi,300); C=np.vstack([np.cos(th),np.sin(th)]); E=A@C
fig,(a,b)=plt.subplots(1,2,figsize=(6.5,3))
a.plot(C[0],C[1],'k'); b.plot(E[0],E[1],'k')
for j,c in enumerate(['b','r']):
    v=Vh[j]; a.annotate("",v,(0,0),arrowprops=dict(color=c,arrowstyle='->')); a.text(*(1.1*v),f"v{j+1}",color=c)
    u=s[j]*U[:,j]; b.annotate("",u,(0,0),arrowprops=dict(color=c,arrowstyle='->')); b.text(*(1.1*u),f"s{j+1} u{j+1}",color=c)
for ax in (a,b): ax.set_aspect('equal'); ax.set_xlabel("x (dimensionless)"); ax.set_ylabel("y (dimensionless)")
a.set_title("unit circle"); b.set_title("image under A")
plt.savefig("least-squares_svd.pdf", bbox_inches="tight")
