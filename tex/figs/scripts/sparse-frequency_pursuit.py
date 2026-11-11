import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=np.array([1,0.]); b=np.array([0.5,np.sqrt(3)/2]); y=a+0.8*b
plt.figure(figsize=(6,4))
for v,l,c in [(a,"a","C0"),(b,"b","C1"),(y,"y","k")]:
    plt.arrow(0,0,*v,head_width=0.03,length_includes_head=True,color=c); plt.text(*(v*1.05),l)
p1=1.4*a; plt.plot([0,p1[0]],[0,p1[1]],"C2--",label="plain step 1: 1.4 a")
p2=p1+0.6*b; plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"C3--",label="plain step 2: +0.6 b")
plt.plot(*p2,"o",mfc="none",ms=9,c="C3"); plt.plot(*y,"ks",label="orthogonal: a+0.8 b")
plt.gca().set_aspect("equal"); plt.xlabel("component 1 (dimensionless)"); plt.ylabel("component 2 (dimensionless)"); plt.legend(fontsize=8)
plt.savefig("sparse-frequency_pursuit.pdf", bbox_inches="tight")
