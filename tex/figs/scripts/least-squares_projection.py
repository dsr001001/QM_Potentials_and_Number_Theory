import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a1=np.array([1,0,1.]); a2=np.array([0,1,1.]); y=np.array([1,2,0.]); yh=np.array([0,1,1.])
fig=plt.figure(figsize=(6,4)); ax=fig.add_subplot(111,projection='3d')
s=np.linspace(-0.5,1.5,10); S,T=np.meshgrid(s,s); P=S[...,None]*a1+T[...,None]*a2
ax.plot_surface(P[...,0],P[...,1],P[...,2],alpha=0.25,color='gray')
for v,c,l in [(a1,'g','a1'),(a2,'m','a2'),(y,'k','y'),(yh,'b','y-hat')]:
    ax.quiver(0,0,0,*v,color=c,arrow_length_ratio=0.1); ax.text(*(v*1.05),l)
ax.plot(*zip(yh,y),'r--'); ax.text(*((y+yh)/2),"residual",color='r')
ax.set_xlabel("component 1"); ax.set_ylabel("component 2"); ax.set_zlabel("component 3")
plt.savefig("least-squares_projection.pdf", bbox_inches="tight")
