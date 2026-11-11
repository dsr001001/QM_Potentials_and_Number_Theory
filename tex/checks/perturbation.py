import numpy as np, qmnt
from scipy.integrate import trapezoid
pi=np.pi
def box(L,Nb):
    n=np.arange(1,Nb+1); E0=(n*pi/(2*L))**2
    D=np.abs(n[:,None]-n[None,:]); S=n[:,None]+n[None,:]
    cm=lambda q: 0.5*(-1)**q*((D==2*q)*1.0-(S==2*q)*1.0)
    xm=np.where(S%2==1,-16*L*np.outer(n,n)/(pi**2*np.where(D==0,1,(n[:,None]**2-n[None,:]**2))**2),0.0)
    return n,E0,cm,xm
# overlap figure integrals (1/2 and 0)
xx=np.linspace(-1,1,20001); rho=np.sin(3*pi*(xx+1)/2)**2
assert abs(trapezoid(rho*np.cos(3*pi*xx),xx)-0.5)<1e-6 and abs(trapezoid(rho*np.cos(2*pi*xx),xx))<1e-6
# RE3 sample
L=44.883; u=(pi/(2*L))**2
assert abs(u-1.22483e-3)<1e-8; assert abs(9*u-0.0110235)<1e-7; assert abs(7*u-0.0085738)<1e-7
assert abs(5e-4/(7*u)-0.058)<1e-3
s2=(5e-4)**2/(9*u-81*u); assert abs(s2+2.83e-6)<1e-8
n,E0,cm,xm=box(L,200); Ee=np.linalg.eigvalsh(np.diag(E0)+1e-3*cm(3))
assert abs(Ee[2]-(E0[2]+5e-4+s2))<1e-7
# three cosines
n,E0,cm,xm=box(1.0,100); c=[0.005,0.01,0.05]
Ee=np.linalg.eigvalsh(np.diag(E0)+c[0]*np.eye(100)+c[1]*cm(1)+c[2]*cm(2))
for i,t in enumerate([0.01,-0.02,0.005,0.005]): assert abs(Ee[i]-E0[i]-t)<1e-4
# Wu-Sprung claims
from scipy.linalg import eigh_tridiagonal
N=3000;h=2*L/(N+1);x=-L+h*np.arange(1,N+1);V=np.array([qmnt.wu_sprung_V(xv) for xv in x])
E,p=eigh_tridiagonal(2/h**2+V,-np.ones(N-1)/h**2,select='i',select_range=(0,99));p/=np.sqrt(h)
A=(p**2*h).T@np.cos(np.outer(x,np.arange(100))*pi/L)
assert abs(E[19]-77)<1
xt=qmnt.wu_sprung_x(E[19]); assert abs(xt-5.4)<0.05 and xt/L<1/8
assert np.allclose(A[:,0],1,atol=1e-8)
assert A[19,1]>0.9; assert np.sum(np.abs(A[19,1:])>0.1)>=3
assert abs(qmnt.wu_sprung_V(0)-3.10073*pi)<1e-6
# tilt
assert abs(-32/(9*pi**2)+0.36025)<1e-5; assert abs(4/243-0.016461)<1e-6
m=np.arange(2,20001,2.0); S=np.sum(m**2/(m**2-1)**5)
assert abs(S-0.016483)<1e-6; assert abs(S-(15-pi**2)*pi**2/3072)<1e-10
assert abs(16/15**5-2.1e-5)<1e-7; assert abs(1024/pi**6-1.06513)<1e-5
E2=-1024/pi**6*S; assert abs(E2+0.017556)<1e-6; assert abs(pi**2/4+E2-2.449845)<1e-6
n,E0,cm,xm=box(1.0,200); ex=np.linalg.eigvalsh(np.diag(E0)+xm)[0]
assert abs(ex-2.44987)<1e-5; assert abs(ex-(pi**2/4+E2))<3e-5
# cos ground state
e60=lambda c: np.linalg.eigvalsh(np.diag(box(1.,60)[1])+c*box(1.,60)[2](1))[0]
ex=e60(1.0); assert abs(ex-2.954413)<1e-6
assert abs(np.linalg.eigvalsh(np.diag(E0)+cm(1))[0]-ex)<1e-10
assert abs(pi**2/4+0.5-2.967401)<1e-6; assert abs(pi**2/4+0.5-1/(8*pi**2)-2.954736)<1e-6
alt=pi**2/4+0.5+0.25/(pi**2/4+0.5-9*pi**2/4); assert abs(alt-2.954407)<1e-6 and abs(alt-ex)<1e-5
Lf,Nf=1.0,4000; xf=qmnt.grid(Lf,Nf); Ef=qmnt.fd_levels(np.cos(pi*xf/Lf),Lf,k=1)
assert abs(Ef[0]-ex)<1e-4
# two-level
assert abs(2-np.sqrt(1.25)-0.881966)<1e-6 and abs(2+np.sqrt(1.25)-3.118034)<1e-6
assert abs(np.sqrt(1.25)-1-0.125+0.0070)<1e-3
# Newton
n60,E060,cm60,_=box(1.0,60); C1=cm60(1)
def g(c):
    w,v=np.linalg.eigh(np.diag(E060)+c*C1); return w[0],v[:,0]@C1@v[:,0]
c1=(3-pi**2/4)/0.5; assert abs(c1-1.065198)<1e-6
En,A1=g(c1); assert abs(En-2.985239)<1e-6 and abs(A1-0.4719)<1e-4
c2=c1+(3-En)/A1; assert abs(c2-1.0965)<1e-4; assert abs(g(c2)[0]-3)<5e-5
assert 0.02<(c2-c1)/c2<0.035
# validity n=1, coverage
assert abs(4*pi**2-39.5)<0.05; assert abs(2*pi**2-19.7)<0.05
for M,v in [(300,110),(400,196),(600,441),(900,992)]: assert abs((M*pi/(2*L))**2-v)<1
# mean spacing levels 201-300
assert abs(np.mean(qmnt.mean_gap(np.arange(201,301)))-1.46)<0.03
# Problems
a12=(32/(9*pi**2))**2/(3*pi**2/4); assert abs(a12-0.0175)<1e-4  # P14
n,E0,cm,xm=box(1.0,200); full=sum(xm[1,j]**2/(E0[1]-E0[j]) for j in range(200) if j!=1)
assert abs(full-0.0052)<1e-4  # P8 full
four=sum(xm[1,j]**2/(E0[1]-E0[j]) for j in [0,2,4,6]); assert abs(four-0.0052)<1e-4 and abs(four-full)<1e-4  # P8 m=1,3,5,7
assert abs(-1/(12*pi**2)+0.0084434)<1e-7  # P7
assert abs(1-np.sqrt(1.01)+0.0049876)<1e-7  # P5
err1=abs(pi**2/4+0.5-1/(8*pi**2)-e60(1.0)); err2=abs(pi**2/4+0.25-0.25/(8*pi**2)-e60(0.5))
assert abs(err1-3.2e-4)<1e-5 and 6<err1/err2<10  # P15
assert abs(-2*1024/pi**6*S+0.035112)<1e-6  # P21
hh=2*44.8827/6001; assert abs(hh-0.014958)<1e-6 and abs(1/hh-66.85)<0.01  # P23
c=0.0; it=0
while True:
    En,A1=g(c)
    if abs(En-3.5)<1e-10 or it>50: break
    c+= (3.5-En)/A1; it+=1
assert it<6 and abs((3.5-pi**2/4)/0.5-2.065198)<1e-6  # P25
print("all checks passed")
