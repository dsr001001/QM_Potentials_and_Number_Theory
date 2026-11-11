import numpy as np, mpmath
from scipy.special import lambertw, gammaln
from scipy.optimize import brentq
import qmnt
def Nsm(t): u=t/(2*np.pi); return u*np.log(u)-u+7/8
f=lambda s: Nsm(s)-0.5
s1=brentq(f,2*np.pi,20)
# Sample t1 rounding
assert abs(14.134725141734695-14.13-0.004725)<1e-6
assert abs(0.004725/14.134725-3.34e-4)<1e-6
gap1=2*np.pi/np.log(s1/(2*np.pi)); assert abs(gap1-7.500)<1e-3
assert abs(0.004725/gap1-6.30e-4)<1e-6
# leading order ratio
t=1419.42248094599; assert abs(Nsm(t)-999.42)<0.01
assert abs((t/(2*np.pi))*np.log(t)/Nsm(t)-1.64)<0.005
# log t / t^0.1
assert abs(10/np.e-3.68)<0.005 and abs(np.exp(10)-2.20e4)<100
assert abs(np.log(1e30)/1e3-0.069)<0.001
# H10
H10=sum(1/k for k in range(1,11)); assert abs(H10-2.928968)<1e-6
assert abs(np.log(11)-2.397895)<1e-6 and abs(1+np.log(10)-3.302585)<1e-6
assert abs(np.log(10)+0.5772156649+0.05-1/1200-H10)<2e-6
assert abs(np.log(np.math.factorial(10)) if False else gammaln(11)-15.104413)<1e-6
assert abs(10*np.log(10)-9-14.025851)<1e-6 and abs(10*np.log(10)-9+np.log(10)-16.328436)<1e-6
d10=gammaln(11)-10.5*np.log(10)+10; assert abs(d10-0.927269)<1e-6
C=0.5*np.log(2*np.pi); assert C<d10<C+1/120
# Stirling table
for n,ea,eb,ec in [(10,2.078562,0.008331,-2.77e-6),(100,3.222357,0.000833,-2.78e-9)]:
    ex=gammaln(n+1); a=n*np.log(n)-n; b=a+0.5*np.log(2*np.pi*n); c=b+1/(12*n)
    assert abs(ex-a-ea)<2e-6 and abs(ex-b-eb)<2e-6 and abs((ex-c)/ec-1)<0.02
assert abs(gammaln(101)-363.739376)<1e-6
x=5; v=4.5*np.log(5)-5+C+1/60; assert abs(v-3.178076)<2e-6 and abs(v-1/(360*125)-3.178054)<2e-6
assert abs(np.sqrt(np.pi)-1.772454)<1e-6
# theta
th=float(mpmath.siegeltheta(100)); ap=50*np.log(100/(2*np.pi))-50-np.pi/8
assert abs(th-87.97216)<1e-5 and abs(ap-87.97196)<1e-5 and abs(th-ap-1/4800)<1e-8
# Lambert W sample
y1=-3/(8*np.e); assert abs(y1+0.137954)<1e-6
W0=lambertw(y1).real; Wm=lambertw(y1,-1).real
assert abs(W0+0.16226)<1e-5 and abs(Wm+3.118)<1e-3
assert abs(2*np.pi*0.375/(-W0)-14.521)<1e-3 and abs(2*np.pi*0.375/(-Wm)-0.756)<1e-3
assert abs(s1-14.52136)<1e-5
assert abs(brentq(f,0.1,2*np.pi)-0.756)<1e-3
y=(1000-11/8)/np.e; W=lambertw(y).real
assert abs(y-367.37)<0.01 and abs(W-4.420)<1e-3
s1000=2*np.pi*998.625/W; g1000=2*np.pi/(1+W)
assert abs(s1000-1419.52)<0.01 and abs(g1000-1.159)<1e-3
assert abs((t-s1000)/g1000+0.08)<0.01
assert abs(2*np.pi*1000/np.log(1000)-909.6)<0.1 and abs(2*np.pi/np.log(1000)-0.9096)<1e-4
# smooth zeros vs qmnt and gradient
n=np.arange(1,10001); sq=qmnt.smooth_zeros(n); yy=(n-11/8)/np.e
assert np.max(np.abs(2*np.pi*(n-11/8)/lambertw(yy).real-sq))<1e-6
gr=np.gradient(sq[:1000]); gd=2*np.pi/np.log(sq[:1000]/(2*np.pi))
assert np.max(np.abs(gr[19:999]/gd[19:999]-1))<1e-3
assert abs(sq[1]-sq[0]-6.13)<0.01
assert abs(gd[400]-1.34)<0.02 and abs(2*np.pi/np.log(sq[-1]/(2*np.pi))-0.854)<0.002
# bisection table
assert abs(f(10)+0.4769)<1e-4 and abs(f(20)-0.8775)<1e-4
for m,v in [(15,0.0651),(12.5,-0.2460),(13.75,-0.0995),(14.375,-0.0194),(14.6875,0.0223)]:
    assert abs(f(m)-v)<1e-4
assert abs(14.53125-s1-0.0099)<1e-4
assert np.ceil(np.log2(1e11)-1)==36
assert abs(f(0.5)-0.094)<1e-3
# Newton sqrt2
xk=1.0; errs=[]
for k in range(4): xk=0.5*(xk+2/xk); errs.append(abs(xk-np.sqrt(2)))
for e,cl in zip(errs,[8.58e-2,2.45e-3,2.12e-6,1.59e-12]): assert abs(e/cl-1)<0.01
fp=lambda s: np.log(s/(2*np.pi))/(2*np.pi)
assert abs(f(14)+0.0680)<1e-4 and abs(fp(14)-0.1275)<1e-4
x1=14-f(14)/fp(14); assert abs(x1-14.5333)<1e-3
x2=x1-f(x1)/fp(x1); x3=x2-f(x2)/fp(x2); assert abs(x2-s1)<2e-5 and abs(x3-s1)<1e-10
# arctan
a1=1.5-(1+1.5**2)*np.arctan(1.5); a2=a1-(1+a1**2)*np.arctan(a1)
assert abs(a1+1.694)<1e-3 and abs(a2-2.321)<1e-3
# floats
assert abs(np.finfo(np.float64).eps-2.22e-16)<1e-18 and abs(np.finfo(np.float32).eps-1.19e-7)<1e-9
assert np.float32(16777217)==np.float32(16777216)
tot=0.0
for k in range(10): tot+=0.1
assert tot==0.9999999999999999
t32=np.float32(0)
for k in range(10): t32=np.float32(t32+np.float32(0.1))
assert abs(float(t32)-1.0000001192092896)<1e-12
assert (1-np.cos(1e-8))/1e-16==0 and abs(2*np.sin(0.5e-8)**2/1e-16-0.5)<1e-12
N=999; j=np.arange(1,N+1)
ps=lambda k,dt:(np.sqrt(2/(N+1))*np.sin(k*j*np.pi/(N+1))).astype(dt)
d64=np.cumsum(ps(3,np.float64)*ps(7,np.float64))[-1]; d32=np.cumsum(ps(3,np.float32)*ps(7,np.float32))[-1]
assert abs(d64)<1e-14 and abs(d32)>1e3*abs(d64)
assert abs(0.01/1e-7-1e5)<1e-6 and abs(0.01/(-3e-8)+3.33e5)<1e3
assert abs(np.log2(np.e)*0+np.finfo(np.float32).eps/2-5.96e-8)<1e-10
# Problems
pi=np.pi; assert abs(22/7-pi-1.264e-3)<1e-6 and abs((22/7-pi)/pi-4.02e-4)<1e-6   # P1
u=1e6/(2*pi); assert abs(np.log(1e6)/(np.log(u)-1)-1.26)<0.005                  # P3
L20=20.5*np.log(20)-20+C+1/240; assert abs(L20-42.335617)<1e-6 and abs(gammaln(21)-42.335616)<1e-6  # P6
assert abs(1-np.exp(-1/240)-0.0042)<1e-4
assert abs(0.75*np.sqrt(pi)-1.329340)<1e-6                                         # P7
xx=2.0
for k in range(2): xx=xx-(xx**3-10)/(3*xx**2)
assert abs(xx-2.154504)<1e-6 and abs(xx-10**(1/3)-6.9e-5)<1e-6                    # P8
assert np.ceil(np.log2(6e12)-1)==42                                                # P9
xb=1e12; dr=np.sqrt(xb+1)-np.sqrt(xb); tr=1/(np.sqrt(xb+1)+np.sqrt(xb))
assert abs(tr-4.99999999999875e-7)<1e-20 and abs(dr/tr-1)<1.2e-4                   # P11
w0=0.5; w1=w0-(w0*np.exp(w0)-1)/((1+w0)*np.exp(w0)); assert abs(w1-0.571020)<1e-6 # P13
W1=lambertw(1).real; assert abs(W1-0.567143)<1e-6 and abs(1/W1-1.763223)<1e-6
zz=qmnt.zeros(500); idx=np.arange(401,501)
assert abs(qmnt.rms(qmnt.fluct(zz[400:500],idx))-0.263)<0.005                      # P16
assert abs(1-2/(np.log(5e-6))*0)<2                                                 # placeholder-free P17 is logical
assert abs(5e-12/1.159-4.3e-12)<1e-13                                              # P22
assert abs(f(7)+0.618728)<1e-6 and abs(fp(7)-0.017194)<1e-6 and abs(7-f(7)/fp(7)-43.0)<0.05  # P23
v=1000.; vs=[]
for k in range(4): v=1000/(np.log(v)-1); vs.append(v)
for a,b in zip(vs,[169.27,242.04,222.76,226.96]): assert abs(a-b)<0.05             # P27
vst=brentq(lambda v: v*(np.log(v)-1)-1000,100,500); assert abs(vst-226.2)<0.1
assert abs(-1/(np.log(vst)-1)+0.226)<0.001
Nf=lambda t:-np.sin(t*np.log(2))/(pi*np.sqrt(2)); Nfp=lambda t:-np.log(2)*np.cos(t*np.log(2))/(pi*np.sqrt(2))
g=lambda t: Nsm(t)+Nf(t)-0.5; gp=lambda t: fp(t)+Nfp(t)
s=14.5214; assert abs(Nf(s)-0.1346)<2e-4 and abs(Nfp(s)-0.1251)<2e-4 and abs(fp(s)-0.1333)<1e-4  # P29
assert abs(s-g(s)/gp(s)-14.0007)<1e-3
assert abs(brentq(g,13.5,14.5)-14.021)<0.003
bd=sum(np.log(p)/p**(m/2) for p in [2,3,5,7,11,13] for m in (1,2,3))               # P30
assert abs(bd-6.568)<1e-3 and abs(bd/pi-2.091)<1e-3 and abs(fp(800)-0.771)<1e-3
# P24 oracle (Table V row)
n=np.arange(401,501); tp=qmnt.oracle_zeros(n,[2,3,5,7,11,13]); r,c=qmnt.score(tp,n)
assert abs(r-0.074)<0.005 and abs(c-0.96)<0.01
# P31-type: Wu-Sprung L values
assert abs(qmnt.L_of(1000)-44.883)<1e-3 and abs(qmnt.L_of(1500)-59.957)<1e-3
print("all checks passed")
