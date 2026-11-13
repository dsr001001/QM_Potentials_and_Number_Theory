import numpy as np, mpmath, inspect
from scipy.special import lambertw, gammaln
from scipy.optimize import brentq
import qmnt
def Nsm(t): u=t/(2*np.pi); return u*np.log(u)-u+7/8
f=lambda s: Nsm(s)-0.5
s1=brentq(f,2*np.pi,20)
assert abs(s1-14.5213)<1e-4   # s_1 quoted throughout
# Sample t1 rounding
assert abs(14.134725141734695-14.13-0.004725)<1e-6
assert abs(0.004725/14.134725-3.34e-4)<1e-6
gap1=2*np.pi/np.log(s1/(2*np.pi)); assert abs(gap1-7.500)<1e-3
assert abs(0.004725/gap1-6.30e-4)<1e-6
# leading order ratio
t=1419.42248094599; assert abs(Nsm(t)-999.42)<0.01
assert abs((t/(2*np.pi))*np.log(t)/Nsm(t)-1.64)<0.005
# log t / t^0.1 pitfall
assert abs(10/np.e-3.68)<0.005 and abs(np.exp(10)-2.20e4)<100
assert abs(np.log(1e30)/1e3-0.069)<0.001
# H10
H10=sum(1/k for k in range(1,11)); assert abs(H10-2.928968)<1e-6
assert abs(np.log(11)-2.397895)<1e-6 and abs(1+np.log(10)-3.302585)<1e-6
assert abs(np.log(10)+0.5772156649+0.05-1/1200-H10)<2e-6
assert abs(H10-np.log(10)-0.626383)<1e-6
assert abs(gammaln(11)-15.104413)<1e-6
assert abs(10*np.log(10)-9-14.025851)<1e-6 and abs(10*np.log(10)-9+np.log(10)-16.328436)<1e-6
d10=gammaln(11)-10.5*np.log(10)+10; assert abs(d10-0.927269)<1e-6
C=0.5*np.log(2*np.pi); assert C<d10<C+1/120
assert abs(C-0.918939)<1e-6 and abs(C+1/120-0.927272)<1e-6
# Stirling table
for n,ea,eb,ec in [(10,2.078562,0.008331,-2.77e-6),(100,3.222357,0.000833,-2.78e-9)]:
    ex=gammaln(n+1); a=n*np.log(n)-n; b=a+0.5*np.log(2*np.pi*n); c=b+1/(12*n)
    assert abs(ex-a-ea)<2e-6 and abs(ex-b-eb)<2e-6 and abs((ex-c)/ec-1)<0.02
assert abs(gammaln(101)-363.739376)<1e-6
v=4.5*np.log(5)-5+C+1/60; assert abs(v-3.178076)<2e-6 and abs(v-1/(360*125)-3.178054)<2e-6
assert abs(np.sqrt(np.pi)-1.772454)<1e-6
# theta
th=float(mpmath.siegeltheta(100)); ap=50*np.log(100/(2*np.pi))-50-np.pi/8
assert abs(th-87.97216)<1e-5 and abs(ap-87.97196)<1e-5 and abs(th-ap-1/4800)<1e-8
# Lambert W sample
y1=-3/(8*np.e); assert abs(y1+0.137954)<1e-6
W0=lambertw(y1).real; Wm=lambertw(y1,-1).real
assert abs(W0+0.16226)<1e-5 and abs(Wm+3.118)<1e-3
assert abs(2*np.pi*0.375/(-W0)-14.521)<1e-3 and abs(2*np.pi*0.375/(-Wm)-0.756)<1e-3
assert abs(2*np.pi/(1+W0)-7.500)<1e-3
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
g1e4=2*np.pi/np.log(sq[-1]/(2*np.pi))
assert abs(gd[400]-1.34)<0.02 and abs(g1e4-0.854)<0.002
assert abs(2*np.pi/gd[400]-4.7)<0.1 and abs(2*np.pi/g1e4-7.4)<0.06   # paper connection remark
assert abs(2*np.pi/np.log(1e4)-0.682)<5e-4   # P21
# bisection table
assert abs(f(10)+0.4769)<1e-4 and abs(f(20)-0.8775)<1e-4
for m,vv in [(15,0.0651),(12.5,-0.2460),(13.75,-0.0995),(14.375,-0.0194),(14.6875,0.0223)]:
    assert abs(f(m)-vv)<1e-4
assert abs(14.53125-s1-0.0099)<1e-4
assert np.ceil(np.log2(1e11)-1)==36
assert abs(f(0.5)-0.094)<1e-3
assert abs(1/2**11-4.9e-4)<1e-5
# Newton sqrt2
xk=1.0; errs=[]
for k in range(4): xk=0.5*(xk+2/xk); errs.append(abs(xk-np.sqrt(2)))
for e,cl in zip(errs,[8.58e-2,2.45e-3,2.12e-6,1.59e-12]): assert abs(e/cl-1)<0.01
fp=lambda s: np.log(s/(2*np.pi))/(2*np.pi)
assert abs(f(14)+0.0680)<1e-4 and abs(fp(14)-0.1275)<1e-4
x1=14-f(14)/fp(14); assert abs(x1-14.5333)<1e-3
x2=x1-f(x1)/fp(x1); x3=x2-f(x2)/fp(x2); assert abs(x2-s1)<2e-5 and abs(x3-s1)<1e-10
# checkpoint Newton
assert abs(0.04*(1e-3)**2-4e-8)<1e-20 and abs(0.04*(4e-8)**2-6e-17)<1e-17
# arctan
a1=1.5-(1+1.5**2)*np.arctan(1.5); a2=a1-(1+a1**2)*np.arctan(a1)
assert abs(a1+1.694)<1e-3 and abs(a2-2.321)<1e-3
# brentq default tolerances
sig=inspect.signature(brentq)
assert sig.parameters['xtol'].default==2e-12
assert abs(sig.parameters['rtol'].default-8.88e-16)<1e-18
assert abs(4*np.finfo(np.float64).eps-8.88e-16)<1e-18
# floats
assert abs(np.finfo(np.float64).eps-2.22e-16)<1e-18 and abs(np.finfo(np.float32).eps-1.19e-7)<1e-9
assert abs(np.finfo(np.float32).eps/2-5.96e-8)<1e-10 and abs(np.finfo(np.float64).eps/2-1.11e-16)<1e-18
assert abs(53*np.log10(2)-15.95)<0.01 and abs(24*np.log10(2)-7.22)<0.01
assert abs(np.finfo(np.float64).max-1.797e308)<1e306 and abs(float(np.finfo(np.float32).max)-3.40e38)<1e36
assert np.float32(16777217)==np.float32(16777216)
assert np.float32(1)+np.float32(1e-8)==np.float32(1) and 1.0+1e-16==1.0 and 1.0+1e-8!=1.0
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
# Problems
pi=np.pi; assert abs(22/7-pi-1.264e-3)<1e-6 and abs((22/7-pi)/pi-4.02e-4)<1e-6   # P1
u=1e6/(2*pi); assert abs(np.log(1e6)/(np.log(u)-1)-1.26)<0.005                  # P3
assert abs(np.log(1e6)-13.815511)<1e-6 and abs(np.log(u)-11.977634)<1e-6
L20=20.5*np.log(20)-20+C+1/240; assert abs(L20-42.335617)<1e-6 and abs(gammaln(21)-42.335616)<1e-6  # P6
assert abs(20.5*np.log(20)-61.412512)<1e-6 and abs(1-np.exp(-1/240)-0.0042)<1e-4
assert abs(0.75*np.sqrt(pi)-1.329340)<1e-6                                         # P7
xx=2.0
for k in range(2): xx=xx-(xx**3-10)/(3*xx**2)
assert abs(xx-2.154504)<1e-6 and abs(xx-10**(1/3)-6.9e-5)<1e-6                    # P8
assert np.ceil(np.log2(6e12)-1)==42 and abs(np.log2(6e12)-42.45)<0.01             # P9
xb=1e12; dr=np.sqrt(xb+1)-np.sqrt(xb); tr=1/(np.sqrt(xb+1)+np.sqrt(xb))
assert abs(tr-4.99999999999875e-7)<1e-20 and abs(dr/tr-1)<1.2e-4                   # P11
assert abs(2.0**-33-1.16e-10)<1e-12
w0=0.5; w1=w0-(w0*np.exp(w0)-1)/((1+w0)*np.exp(w0)); assert abs(w1-0.571020)<1e-6 # P13
W1=lambertw(1).real; assert abs(W1-0.567143)<1e-6 and abs(1/W1-1.763223)<1e-6
zz=qmnt.zeros(500); idx=np.arange(401,501)
assert abs(qmnt.rms(qmnt.fluct(zz[400:500],idx))-0.263)<0.005                      # P16
assert abs(0.01/(-3e-8)+3.33e5)<1e3                                                # P19
assert abs(5e-12/1.159-4.3e-12)<1e-13                                              # P22
assert abs(f(7)+0.61873)<1e-5 and abs(fp(7)-0.017194)<2e-6                         # P23
step=-f(7)/fp(7); assert abs(step-35.985)<1e-3 and abs(7+step-43.0)<0.05
vv=1000.; vs=[]
for k in range(4): vv=1000/(np.log(vv)-1); vs.append(vv)
for a,b in zip(vs,[169.27,242.04,222.76,226.96]): assert abs(a-b)<0.05             # P27
assert abs(np.log(1000)-1-5.907755)<1e-6
vst=brentq(lambda v: v*(np.log(v)-1)-1000,100,500); assert abs(vst-226.2)<0.1
assert abs(-1/(np.log(vst)-1)+0.226)<0.001
Nf=lambda t:-np.sin(t*np.log(2))/(pi*np.sqrt(2)); Nfp=lambda t:-np.log(2)*np.cos(t*np.log(2))/(pi*np.sqrt(2))
g=lambda t: Nsm(t)+Nf(t)-0.5; gp=lambda t: fp(t)+Nfp(t)
s=14.5214; assert abs(Nf(s)-0.1346)<2e-4 and abs(Nfp(s)-0.1251)<2e-4 and abs(fp(s)-0.1333)<1e-4  # P29
assert abs(g(s)-0.1346)<2e-4 and abs(gp(s)-0.2584)<2e-4
assert abs(s-g(s)/gp(s)-14.0007)<1e-3 and abs(g(s)/gp(s)-0.5207)<1e-3
root29=brentq(g,13.5,14.5); assert abs(root29-14.021)<0.003
assert abs(s-14.1347-0.387)<1e-3 and abs(root29-14.1347+0.11)<0.01
bd=sum(np.log(p)/p**(m/2) for p in [2,3,5,7,11,13] for m in (1,2,3))               # P30
assert abs(bd-6.568)<1e-3 and abs(bd/pi-2.091)<1e-3 and abs(fp(800)-0.771)<1e-3
# P24 oracle (Table V row)
n=np.arange(401,501); tp=qmnt.oracle_zeros(n,[2,3,5,7,11,13]); r,c=qmnt.score(tp,n)
assert abs(r-0.074)<0.005 and abs(c-0.96)<0.01
# P25 is the theta check above
# P31 Wu-Sprung
V0=qmnt.V0
x1000=qmnt.wu_sprung_x(1000.0); x1500=qmnt.wu_sprung_x(1500.0)
assert abs(x1000-44.883)<1e-3 and abs(x1500-59.957)<1e-3
assert abs(qmnt.L_of(1000)-44.883)<1e-3 and abs(qmnt.L_of(1500)-59.957)<1e-3
assert abs(qmnt.wu_sprung_V(x1000)-1000)<1e-4 and abs(qmnt.wu_sprung_V(x1500)-1500)<1e-4
def xws(V): return (np.sqrt(V-V0)*np.log(V0/(2*np.pi*np.e**2))+np.sqrt(V)*np.log((np.sqrt(V)+np.sqrt(V-V0))/(np.sqrt(V)-np.sqrt(V-V0))))/np.pi
assert abs(xws(1000.0)-x1000)<1e-6 and abs(xws(V0))<1e-12
Vg=np.linspace(V0*1.0001,1500,2000); xg=np.array([qmnt.wu_sprung_x(vv) for vv in Vg])
assert np.all(np.diff(xg)>0)
# ---- revision additions ----
assert abs((0.01/0.002)/(0.01/100)-5e4)<1e-6                     # football vs coin
assert round(22/7,3)==3.143 and round(np.pi,3)==3.142             # P1 answer
assert abs((22/7-np.pi)/np.pi-4.0e-4)<5e-6
aa,bb=-1.0,2.0; mids=[]                                           # bisection on 1/x over [-1,2]
for k in range(3):
    mm=(aa+bb)/2; mids.append(mm)
    if (1/aa)*(1/mm)<=0: bb=mm
    else: aa=mm
assert mids==[0.5,-0.25,0.125]
yR=(1000-11/8)/np.e; WR=lambertw(yR).real                        # t_1000 sample
assert abs(yR-367.374)<1e-3 and abs(WR-4.4202)<6e-5
assert abs(2*np.pi*998.625-6274.546)<1e-3 and abs(6274.546/4.4202-1419.52)<0.005
assert abs(6274.546/4.420-1419.58)<0.005
assert abs((1419.42248094599-6274.546/4.420)/1.159+0.14)<0.01
assert abs((1419.42248094599-1419.52)/1.159+0.08)<0.01
assert 1e-4/4.3e-12>1e7                                           # precision paragraph, P22
assert 1000/3<=385<=1330/3 and sum(k*k for k in range(1,11))==385  # P5 instance n=10
x0=1.0; x1b=0.5*(x0+2/x0)                                         # P15(b) instance a=2
assert abs((x1b-np.sqrt(2))-(x0-np.sqrt(2))**2/(2*x0))<1e-12
p17a=lambda x:(x-1)*(x-2); assert p17a(0)==2 and p17a(3)==2       # P17
assert 1/(-1)==-1 and 1/2==0.5
p17d=lambda x:(x-1)*(x-2)*(x-3); assert p17d(0)==-6 and p17d(4)==6 and p17d(2)==0
assert abs(Nsm(2*np.pi)-0.5+0.625)<1e-12                          # safe bracket sample
print('all checks passed')
