import numpy as np
from scipy.linalg import eig, eigh_tridiagonal
# Sample: leaky level
L=44.883; E10=(10*np.pi/(2*L))**2
assert abs(E10-0.1225)<5e-4
assert abs(np.exp(-1)-0.3679)<1e-4
assert abs(np.log(2)/0.02-34.66)<0.01
# Sample: complex symmetric 2x2 (sort by imaginary part; real parts are equal)
l=sorted(np.linalg.eigvals(np.array([[1,2j],[2j,3]])),key=lambda z:z.imag)
assert np.allclose(l,[2-np.sqrt(3)*1j,2+np.sqrt(3)*1j])
assert abs(np.sqrt(3)-1.7321)<1e-4
# Sample: two levels
def shift(a,D=1.0):
    d=np.sqrt(complex(D**2-4*a**2)); return (0.5*D-0.5*d).real  # lambda_- - E1 with E1=0
assert abs(shift(0.1)-0.010102)<1e-6
assert abs(shift(0.4)-0.2)<1e-9
assert abs(shift(1.0)-0.5)<1e-9
assert abs(np.sqrt(3)/2-0.8660)<1e-4
# Sample: HO with i eps x, numerical
Lb,N=10.0,3000; h=2*Lb/(N+1); x=-Lb+h*np.arange(1,N+1); eps=0.6
from scipy.sparse import diags; from scipy.sparse.linalg import eigs
Hm=diags([np.full(N-1,-1/h**2),2/h**2+x**2+1j*eps*x,np.full(N-1,-1/h**2)],[-1,0,1]).tocsc()
ev=eigs(Hm,k=1,sigma=1.0,return_eigenvectors=False)[0]
assert abs(ev.real-1.09)<2e-3 and abs(ev.imag)<1e-6
assert abs(1-0.6**2/4-0.91)<1e-12
# box sine elements and four-state sample
def s1(a,b):
    if (a+b)%2==0: return 0.0
    return -4/np.pi*(1/(4-(a-b)**2)-1/(4-(a+b)**2))
assert abs(s1(1,2)+0.6791)<1e-4 and abs(s1(1,4)-0.1940)<1e-4
assert abs(s1(2,3)+0.4850)<1e-4 and abs(s1(3,4)+0.4527)<1e-4
def Tn(K):
    n=np.arange(1,K+1); return np.array([sum(s1(a,b)**2/(a*a-b*b) for b in n if b!=a) for a in n])
T4=Tn(4)
assert np.allclose(T4,[-0.1562,0.1067,0.0178,0.0318],atol=2e-4)
assert abs(T4.sum())<1e-14
# brute check of sine element formula by quadrature
Lp=np.pi/2; xx=np.linspace(-Lp,Lp,200001)
ps=lambda k: np.sin(k*np.pi*(xx+Lp)/(2*Lp))/np.sqrt(Lp)
assert abs(np.trapezoid(ps(1)*ps(2)*np.sin(np.pi*xx/Lp),xx)-s1(1,2))<1e-6
# Problem 17
assert abs(s1(4,5)+0.4410)<1e-4 and abs(s1(2,5)-0.2264)<1e-4
T5=Tn(5); assert abs(T5[3]-0.0102)<3e-4 and abs(T5.sum())<1e-14
# width sample: cos element check
def c1(a,b): return -0.5*(int(abs(a-b)==2)-int(a+b==2))
g=0.01; A11=-g*(1+c1(1,1)); assert abs(-2*A11-0.03)<1e-12 and abs(-2*(-g*(1+c1(5,5)))-0.02)<1e-12
assert abs(1/0.03-33.3)<0.05 and abs(1/0.02-50)<1e-12
assert abs(np.trapezoid(ps(1)**2*np.cos(np.pi*xx/Lp),xx)-0.5)<1e-6
# null mode toy
assert abs(np.sqrt(0.01/1e-14)-1e6)<1e-3
# locality 89% (partners m<=400)
n=20; w=np.array([s1(n,m)**2/abs(n*n-m*m) for m in range(1,401) if m!=n])
frac=(s1(20,19)**2/39+s1(20,21)**2/41)/w.sum(); assert abs(frac-0.89)<0.01
# two-level estimate of PT breaking threshold
assert abs(1.5/0.6791-2.2)<0.05
# complex plane claims
K=60; nn=np.arange(1,K+1); E=nn**2.0
S=np.array([[s1(a,b) for b in nn] for a in nn])
assert np.abs(eig(np.diag(E)+1j*S,right=False).imag).max()<1e-8
assert np.abs(eig(np.diag(E)+4j*S,right=False).imag).max()>0.1
Cos=np.eye(K)+np.array([[c1(a,b) for b in nn] for a in nn])
for gg in (0.5,2):
    assert eig(np.diag(E)-1j*gg*Cos,right=False).imag.max()<1e-10
# exact vs 2nd order at c=0.5
dE=E[:,None]-E[None,:]; np.fill_diagonal(dE,np.inf); T=(S**2/dE).sum(1)
l,v=eig(np.diag(E)+0.5j*S); mt=np.argmax(np.abs(v)**2,0); ex=np.zeros(K)
for j in range(K): ex[mt[j]]=l[j].real-E[mt[j]]
assert len(set(mt))==K and np.corrcoef(ex[:50],-0.25*T[:50])[0,1]>0.99
# null-mode overlaps
Lq,Nq=np.pi/2,400; hq=2*Lq/(Nq+1)
_,psi=eigh_tridiagonal(np.full(Nq,2/hq**2),np.full(Nq-1,-1/hq**2),select="i",select_range=(0,59))
psi/=np.sqrt(hq); off=~np.eye(60,dtype=bool)
assert np.abs((psi.T@psi*hq)[off]).max()<1e-11
p32=psi.astype(np.float32); m32=np.abs(((p32.T@p32)*np.float32(hq))[off]).max()
assert 1e-9<m32<1e-5
# problems
assert abs(np.log(2)/0.1-6.931)<1e-3 and abs(np.exp(-2)-0.1353)<1e-4      # P1
assert abs(1/0.6-1.667)<1e-3                                              # P2
assert np.allclose(np.sort(np.linalg.eigvals([[0,1j],[1j,0]]).imag),[-1,1]) # P3
assert abs(shift(0.3)-0.1)<1e-12 and abs(0.3**2-0.09)<1e-12               # P9
Tm=np.array([[1,2],[2,-3.]]); c=np.array([0.5,1]); f=lambda c: c@Tm@c
gr=[(f(c+1e-6*np.eye(2)[k])-f(c-1e-6*np.eye(2)[k]))/2e-6 for k in range(2)]
assert np.allclose(gr,[5,-4],atol=1e-6)                                    # P13
l14=np.linalg.eigvals([[1,1j],[1j,2]]); assert abs(l14.real.sum()-3)<1e-12 # P14
assert abs(1/16-0.0625)<1e-12                                              # P23
assert abs(652*2.2e-16-1.4e-13)<0.05e-13                                   # P29
assert abs(1/100-0.01)<1e-12                                               # P31
print("all checks passed")
