import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
def f(s): u=s/(2*np.pi); return u*np.log(u)-u+7/8-0.5
fp=lambda s: np.log(s/(2*np.pi))/(2*np.pi)
r=brentq(f,10,20,xtol=1e-15)
a,b=10.,20.; eb=[]
for k in range(40):
    m=(a+b)/2; eb.append(abs(m-r))
    if f(a)*f(m)<=0: b=m
    else: a=m
x=10.; en=[]
for k in range(8): x=x-f(x)/fp(x); en.append(abs(x-r))
x0,x1=10.,20.; es=[]
for k in range(10):
    if f(x1)==f(x0): break
    x2=x1-f(x1)*(x1-x0)/(f(x1)-f(x0)); x0,x1=x1,x2; es.append(abs(x2-r))
ev=[]; g=lambda s: (ev.append(s), f(s))[1]; brentq(g,10,20,xtol=1e-15)
er=[abs(v-r) for v in ev]
cl=lambda e: np.maximum(e,1e-16)
plt.figure(figsize=(6,4))
for e,l in [(eb,"bisection"),(en,"Newton"),(es,"secant"),(er,"Brent")]:
    plt.semilogy(range(1,len(e)+1),cl(np.array(e)),'o-',ms=3,label=l)
plt.xlabel("iteration (function evaluation for Brent)"); plt.ylabel("|x_k - s_1|"); plt.legend()
plt.savefig("asymptotics_convergence.pdf", bbox_inches="tight")
