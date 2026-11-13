import numpy as np, qmnt
from scipy.optimize import brentq
from scipy.stats import norm
def Nsm(t): u=np.asarray(t)/(2*np.pi); return u*np.log(u)-u+7/8
# Sample: four-point fit
tt=np.array([0,np.pi/2,np.pi,3*np.pi/2]); yy=np.array([1.1,2.0,-0.9,-2.2])
X=np.c_[np.sin(tt),np.cos(tt)]; a,b=np.linalg.lstsq(X,yy,rcond=None)[0]
assert abs(a-2.1)<1e-9 and abs(b-1.0)<1e-9
assert abs(np.hypot(a,b)-2.326)<1e-3 and abs(np.arctan2(b,a)-0.444)<1e-3
assert abs(np.sqrt(np.mean((yy-X@[a,b])**2))-0.1)<1e-9
# zeros heights, gaps
t=qmnt.zeros(2000)
assert abs(t[399]-680)<3; assert abs(t[399]-t[0]-666)<3
assert abs(t[1999]-2515)<5; assert abs(t[449]-745)<5
assert abs(t[400]-680)<5 and abs(t[499]-811.18)<0.01
assert abs(t[1999]-347-2170)<10
g400=qmnt.mean_gap(np.array([400]))[0]; assert abs(g400-1.34)<0.02
assert abs(g400**2/(4*np.pi*680)-2e-4)<0.5e-4
# Sec 1: relative change of Nsm' over one spacing at zero 400
assert abs(1.34/(2*np.pi*680*0.745)-4e-4)<0.5e-4
# 800-zero window length
assert abs(t[799]-t[0]-1170)<10
# primes
P=qmnt.first_primes(300); assert P[24]==97 and P[149]==863 and P[299]==1987
assert abs(3*np.log(97)-13.724)<1e-3
# C1
dw=8/450; assert abs(2*np.pi/dw-353.4)<0.05; assert abs(dw/(2*np.pi/666)-1.9)<0.05
assert abs(2*np.pi/666-0.0094)<1e-4
assert abs(39*dw-0.693333)<1e-6 and abs(39*dw-np.log(2)-1.86e-4)<1e-6
assert abs(450/8*np.log(2)-38.99)<0.005
assert abs(np.pi/(450.5*dw)-0.39)<0.005 and abs(450.5*dw-8.009)<1e-3
j=np.arange(1,451); assert abs(np.cos(0*j*dw).sum()-450)<1e-9
assert abs(0.274-0.267-0.007)<1e-12 and abs(0.263-0.258-0.005)<1e-12
# periodogram peak
assert abs(400*0.225**2/4-5.06)<0.01; assert abs(1/(np.pi*np.sqrt(2))-0.2251)<1e-4
s=np.sin(np.pi/8)/(np.pi/8); assert abs(s-0.9745)<1e-4 and abs(s**2-0.9496)<1e-4
# aliasing pitfall numbers
assert abs(np.pi/1.34-2.34)<0.01 and abs(np.log(131)-4.88)<0.01
# matching pursuit
assert abs(4/np.sqrt(2)-2.828)<1e-3
bb=np.array([0.5,np.sqrt(3)/2]); y2=np.array([1,0])+0.8*bb
assert abs(y2@bb-1.3)<1e-9; r=y2-1.4*np.array([1,0]); assert abs(r@bb-0.6)<1e-9
assert abs(np.linalg.norm(r-0.6*bb)-0.346)<1e-3
# Answer 9
a9=np.array([1.,0.]); b9=np.array([.6,.8]); y9=.5*a9+b9
assert np.allclose(y9,[1.1,0.8]) and abs(y9@a9-1.1)<1e-12 and abs(y9@b9-1.3)<1e-12
r9=y9-1.3*b9; assert np.allclose(r9,[0.32,-0.24]) and abs(r9@a9-0.32)<1e-12
assert abs(np.linalg.norm(r9-0.32*a9)-0.24)<1e-12
assert np.allclose(np.linalg.solve(np.c_[a9,b9],y9),[0.5,1.0])
# validation bookkeeping
assert len(np.arange(-8,3.01,0.5))==23 and 400//3==133 and 800//3==266
assert abs(7.95*4*666/(2*np.pi)-3400)<50 and abs(7.95*4*1170/(2*np.pi)-5900)<50
# precision
sd=np.sqrt(24)*0.18/(0.225*20*666); assert abs(sd-2.94e-4)<1e-6
assert abs(np.sqrt(24)-4.899)<1e-3 and abs(np.sqrt(24)*0.18-0.8818)<1e-4 and abs(0.225*20*666-2997)<1
assert abs(sd/(2*np.pi/666)-0.031)<0.001 and abs((np.pi/666)/sd-16)<0.5
assert abs(np.sqrt(2)*1170/666-2.5)<0.02
# dephasing
assert abs(2*np.sin(0.3)-0.591)<1e-3 and abs(np.cos(0.6)-0.825)<1e-3
assert abs((np.pi/3)/3e-4-3491)<1; assert abs((14+680)/2-347)<1
# amplitude table (B_q and log q)
for q,p,m,B,Lq in [(2,2,1,.2251,.6931),(3,3,1,.1838,1.0986),(4,2,2,.0796,1.3863),(5,5,1,.1424,1.6094),(7,7,1,.1203,1.9459),(8,2,3,.0375,2.0794),(9,3,2,.0531,2.1972),(11,11,1,.0960,2.3979),(13,13,1,.0883,2.5649),(16,2,4,.0199,2.7726)]:
    assert abs(1/(np.pi*m*np.sqrt(q))-B)<1e-4 and abs(np.log(q)-Lq)<1e-4
# 43 prime powers
pr=qmnt.primes_upto(211); pp=[q for q in range(2,114) if any(q==int(p)**k for p in pr for k in range(1,8))]
assert len([q for q in pp if q in pr])==30 and len(pp)==40
assert len(pp)-3+6==43 and len([p for p in pr if 131<=p<=211])==16
for q,m,B in [(64,6,.0066),(81,4,.0088),(32,5,.0113)]: assert abs(1/(np.pi*m*np.sqrt(q))-B)<1e-4
assert abs(1/(np.pi*np.sqrt(211))-0.0219)<1e-4 and abs(1/(np.pi*np.sqrt(131))-0.0278)<1e-4
# chance density
w=brentq(lambda w:0.0094*np.exp(w)/w-1,3,10); assert abs(w-6.5)<0.15
assert abs(1-np.exp(-1)-0.63)<0.01
# RE2 roots
assert abs(np.log(2)/np.pi-0.2206)<1e-4 and abs(2**-.5+.5+2**-1.5-1.5607)<1e-4
mx=np.log(2)/np.pi*(2**-.5+.5+2**-1.5); assert abs(mx-0.3443)<1e-4
assert abs(2*np.pi*mx-2.1636)<1e-3 and abs(np.exp(2*np.pi*mx)-8.70)<0.01
ts=2*np.pi*np.exp(2*np.pi*mx); assert abs(ts-54.7)<0.2
d=lambda x:np.log(x/(2*np.pi))/(2*np.pi)
assert abs(d(14.134725)-0.129)<1e-3 and abs(d(680)-0.745)<1e-3 and abs(d(t[449])-0.77)<0.02
assert abs(14.134725/(2*np.pi)-2.2496)<1e-3 and abs(680/(2*np.pi)-108.2)<0.05
assert d(t[400])>0.745-1e-3
bound=sum(np.log(p)/np.pi*sum(p**(-m/2) for m in (1,2,3)) for p in P[:25]); assert bound>4
assert abs(0.6*3-1.8)<1e-12
# answers
assert abs(np.arctan2(-0.4,0.3)+0.927)<1e-3
assert abs(np.pi/1170-0.002685)<1e-6 and abs(1.0996-np.log(3))<np.pi/1170
assert abs(2*np.pi/1170-0.00537)<1e-5 and abs(abs(1.0996-np.log(3))-0.000988)<1e-6
assert abs((np.pi/3)/0.002-523.6)<0.1 and len(np.arange(-6,2.01,0.5))==17
assert abs(400*0.184**2/4-3.39)<0.01 and abs(0.225**2/0.184**2-1.50)<0.01
assert abs(400/72-5.56)<0.01 and abs(800/129-6.20)<0.01
assert abs(400/900-0.444)<1e-3 and abs(800/900-0.889)<1e-3
phi1=2*np.arcsin(0.034/0.263/2); phi2=2*np.arcsin(0.078/0.279/2)
assert abs(phi1-0.129)<0.002 and abs(phi2-0.280)<0.002
sd5=np.sqrt(24)*0.2/(0.142*np.sqrt(800)*1170); assert abs(sd5-2.09e-4)<2e-6 and abs(sd5/(np.pi/1170)-0.078)<0.002
assert abs(np.sqrt(24)*0.2-0.980)<1e-3 and abs(0.142*np.sqrt(800)*1170-4699)<1
sd16=np.sqrt(24)*0.2/(0.0199*np.sqrt(800)*1170); assert abs(sd16-1.5e-3)<1e-4 and abs(sd16/(np.pi/1170)-0.55)<0.03
# Answer 23: tolerance in standard deviations and miss probability
z16=(np.pi/1170)/sd16; assert abs(z16-1.8)<0.05 and abs(2*norm.sf(z16)-0.07)<0.01
assert abs((np.pi/1170)/sd5-13)<0.5
c=np.pi/666; assert abs(c-0.004717)<1e-6
assert abs(1.3870-np.log(4))<c and abs(2.3979-np.log(11))<c
assert abs(abs(1.3870-np.log(4))-0.00071)<1e-5 and abs(2.3979-np.log(11))<1e-5
assert abs(1.7-np.log(5))>c and abs(1.7-np.log(7))>c
assert abs(np.log(127/125)/(2*np.pi/666)-1.68)<0.01
# Answer 1 (50 and 100 primes)
assert 50*3==150 and 50*3*2==300 and 100*3==300 and 100*3*2==600
assert 300<400 and 600>400 and 600<800
# Answer 15: prime powers m>=2 up to 200
pr200=[int(p) for p in qmnt.primes_upto(200)]
pw=[(p**m,p,m) for p in pr200 for m in range(2,9) if p**m<=200]
assert len(pw)==14
amp=sorted((1/(np.pi*m*np.sqrt(q)),q) for q,p,m in pw)
assert [q for _,q in amp[:5]]==[128,64,81,125,32]
for (B,q),Bc in zip(amp[:5],[.0040,.0066,.0088,.0095,.0113]): assert abs(B-Bc)<1e-4
assert min(1/(np.pi*np.sqrt(p)) for p in pr200)>0.0113
assert abs(1/(np.pi*np.sqrt(199))-0.0226)<1e-4
# Answer 21: p=3 wave and {2,3} sum
mx3=np.log(3)/np.pi*(3**-.5+1/3+3**-1.5)
assert abs(np.log(3)/np.pi-0.3497)<1e-4 and abs(3**-.5+1/3+3**-1.5-1.1031)<1e-4
assert abs(mx3-0.3858)<1e-4
assert abs(2*np.pi*np.exp(2*np.pi*mx3)-70.9)<0.2
tot=mx+mx3; assert abs(tot-0.7301)<1e-4
assert abs(2*np.pi*np.exp(2*np.pi*tot)-617)<2
assert d(t[400])>tot and t[400]>617
# Problem 17 / greedy first pick: largest peak near log 2
y=np.arange(1,401)-0.5-Nsm(t[:400]); T=t[399]-t[0]
wg=np.arange(0.05,8,0.25*2*np.pi/T)
Pw=(np.cos(np.outer(wg,t[:400]))@y)**2+(np.sin(np.outer(wg,t[:400]))@y)**2
assert abs(wg[np.argmax(Pw)]-np.log(2))<np.pi/T
# Problem 19: six-prime oracle on 401-500
hold=np.arange(401,501); tp=qmnt.oracle_zeros(hold,qmnt.first_primes(6))
rm,co=qmnt.score(tp,hold); assert abs(rm-0.074)<0.003 and abs(co-0.96)<0.01
print("all checks passed")
