import numpy as np
# Sample: rolling cart
t=np.array([0,1,2,3.]); y=np.array([1,3,4,4.])
c1,c0=np.polyfit(t,y,1); assert abs(c1-1)<1e-12 and abs(c0-1.5)<1e-12
r=y-(c0+c1*t); assert abs(np.sum(r**2)-1.0)<1e-12
# Sample: projection
A=np.array([[1,0],[0,1],[1,1.]]); yy=np.array([1,2,0.])
c=np.linalg.lstsq(A,yy,rcond=None)[0]; assert np.allclose(c,[0,1])
assert np.allclose(yy-A@c,[1,1,-1])
# Sample: SVD 2x2
A=np.array([[3,0],[4,5.]]); U,s,Vh=np.linalg.svd(A)
assert np.allclose(s,[3*np.sqrt(5),np.sqrt(5)]); assert abs(s[0]-6.708)<1e-3 and abs(s[1]-2.236)<1e-3
assert abs(s[0]*s[1]-15)<1e-12
# Sample: min norm
assert np.allclose(np.linalg.pinv(np.array([[1,2.]]))@[5],[1,2])
# Sample: ill-conditioned
A=np.array([[1,1],[1,1.001]])
ca=np.linalg.solve(A,[2,2.001]); cb=np.linalg.solve(A,[2,2.002])
assert np.allclose(ca,[1,1]) and np.allclose(cb,[0,2])
rel_y=0.001/np.linalg.norm([2,2.001]); amp=1/rel_y
assert abs(rel_y-3.535e-4)<1e-6 and abs(amp-2829)<2
sv=np.linalg.svd(A,compute_uv=False); kap=sv[0]/sv[1]
assert abs(kap-4002)<2 and abs(sv[0]-2.0005)<1e-4 and abs(sv[1]-4.999e-4)<1e-7
# Sample: ridge vs truncation
for al,c1e,c2e,err in [(1e-5,0.99999,1.8182,0.818),(1e-4,0.99990,1.0000,1.0e-4),(1e-3,0.99900,0.1818,0.818)]:
    a1=1/(1+al); a2=2e-4/(1e-4+al)
    assert abs(a1-c1e)<1e-5 and abs(a2-c2e)<1e-4
    assert abs(np.hypot(a1-1,a2-1)-err)<(1e-3 if err>0.01 else 2e-5)
# RE4 table and fits
tj=0.7*np.arange(10); eps=np.array([.15,-.12,.10,-.15,.12,-.10,.14,-.13,.11,-.14])
tab_sin=[0,0.6442,0.9854,0.8632,0.3350,-0.3508,-0.8716,-0.9825,-0.6313,0.0168]
assert np.allclose(np.sin(tj),tab_sin,atol=6e-5)
yj=np.sin(tj)+eps
assert np.allclose(yj,[0.15,0.5242,1.0855,0.7132,0.4550,-0.4508,-0.7316,-1.1125,-0.5213,-0.1232],atol=6e-5)
u=lambda x: x/3.15-1
k_t=np.linalg.cond(np.vander(tj,10)); k_u=np.linalg.cond(np.vander(u(tj),10))
assert k_t/k_u>100 and k_t>1e6  # text: ratio exceeds 100
tt=np.linspace(0,6.3,2000); tr=[];te=[]
for d in range(10):
    cc=np.linalg.lstsq(np.vander(u(tj),d+1),yj,rcond=None)[0]
    tr.append(np.sqrt(np.mean((np.vander(u(tj),d+1)@cc-yj)**2)))
    te.append(np.sqrt(np.mean((np.vander(u(tt),d+1)@cc-np.sin(tt))**2)))
assert all(tr[i+1]<=tr[i]+1e-12 for i in range(9)); assert tr[9]<1e-10
assert te[3]<te[1] and te[9]>2*te[3]
# Prob 27
assert abs(6.3**9-1.563e7)<1e5 and 6.3**9/np.sqrt(10)>4.9e6
# Sample: interpolation vs extrapolation
P=np.array([0,np.pi/2,np.pi]); Y=np.array([0,1,0.])
l1,l0=np.polyfit(P,Y,1); assert abs(l1)<1e-12 and abs(l0-1/3)<1e-12
par=lambda x:4/np.pi**2*x*(np.pi-x)
assert abs(par(np.pi/4)-0.75)<1e-12 and abs(par(1.5*np.pi)+3)<1e-12
assert abs(1/3-np.sin(np.pi/4)+0.374)<1e-3 and abs(0.75-np.sin(np.pi/4)-0.043)<1e-3
# Sample: prime-2 wave at RE1 heights (rank 6 listing)
RE1=np.array([14.134725,21.022040,25.010858,30.424876,32.935062,37.586178,40.918719,43.327073,48.005151,49.773832])
L2=np.log(2); F=lambda x:-sum(np.sin(m*x*L2)/(m*2**(m/2)) for m in (1,2,3))/np.pi
des=lambda x,w: np.hstack([np.sin(np.outer(x,w)),np.cos(np.outer(x,w))])
D=des(RE1,L2*np.array([1,2,3.])); assert np.linalg.matrix_rank(D)==6
cc=np.linalg.lstsq(D,F(RE1),rcond=None)[0]
assert np.allclose(cc[:3],[-0.22508,-0.07958,-0.03751],atol=1e-5) and np.allclose(cc[3:],0,atol=1e-8)
# Prob 19: extrapolation errors
ts=np.linspace(10,60,80); te_=np.linspace(60.01,100,1000)
c3=np.linalg.lstsq(des(ts,L2*np.array([1,2,3.])),F(ts),rcond=None)[0]
cg=np.linalg.lstsq(des(ts,8*np.arange(1,61)/60),F(ts),rcond=None)[0]
e3=np.sqrt(np.mean((des(te_,L2*np.array([1,2,3.]))@c3-F(te_))**2))
eg=np.sqrt(np.mean((des(te_,8*np.arange(1,61)/60)@cg-F(te_))**2))
assert e3<1e-10 and eg>0.1*np.sqrt(np.mean(F(te_)**2))
assert np.sqrt(np.mean((des(ts,8*np.arange(1,61)/60)@cg-F(ts))**2))<1e-6
# Sample: damped step
A=np.array([[3,0],[4,5.]]); r=np.array([1,1.])
def step(A,r,lam):
    U,s,Vh=np.linalg.svd(A); return Vh.T@((s/(s**2+lam*s[0]**2))*(U.T@r))
dc=step(A,r,0.01); assert np.allclose(dc,[0.3155,-0.0515],atol=1e-4)
assert np.allclose(np.linalg.solve(A,r),[1/3,-1/15])
# Prob 17
big=step(A,r,1e6); approx=A.T@r/(1e6*45)
assert np.allclose(approx,[1.556e-7,1.111e-7],rtol=1e-3) and np.allclose(big,approx,rtol=1e-5)
# Prob 1, 5, 6, 7, 9, 11
c1,c0=np.polyfit([0,1,2],[0,1,3],1); assert abs(c1-1.5)<1e-12 and abs(c0+1/6)<1e-12
assert abs(np.sum((np.array([0,1,3])-c0-c1*np.array([0,1,2]))**2)-1/6)<1e-12
assert np.allclose(np.linalg.pinv(np.array([[3,4.]]))@[10],[1.2,1.6])
assert abs(1.4-7/5)<1e-15
Ap=np.linalg.pinv(np.array([[1,1],[0,0.]])); assert np.allclose(Ap,[[.5,0],[.5,0]]); assert np.allclose(Ap@[2,3],[1,1])
assert abs(1/(2*np.sqrt(1e-4*100))-5)<1e-12
# Prob 13
n=np.arange(1,11); s1,s0=np.polyfit(n,RE1,1)
assert abs(s1-3.863)<1e-3 and abs(s0-13.07)<1e-2
g=2*np.pi/np.log(RE1.mean()/(2*np.pi)); assert abs(g-3.70)<5e-3
assert abs(s1/g-1.044)<1e-3  # Prob 13 ratio
# Prob 16 count
assert len(np.arange(-8,3.01,0.5))==23
# Prob 21
uu=np.linspace(-1,1,30); Vd=np.vander(uu,6)
assert abs(np.linalg.cond(Vd.T@Vd)/np.linalg.cond(Vd)**2-1)<1e-6
# Prob 24 example: alpha*=1e-4, minimum MSE
sig,ep,x=0.01,0.01,1; a=ep**2/x**2; mse=(a**2*x**2+sig**2*ep**2)/(sig**2+a)**2
assert abs(a-1e-4)<1e-15 and abs(mse-0.5)<1e-12
print('all checks passed')
