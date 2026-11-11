import numpy as np
from scipy.special import ai_zeros
from scipy.integrate import quad
import qmnt
from scipy.linalg import eigh_tridiagonal

# Sample: local wavelength linear well
assert abs(2*np.pi/np.sqrt(10)-1.987)<1e-3
assert abs(2*np.pi/np.sqrt(5)-2.810)<1e-3
assert abs(2*np.pi/np.sqrt(0.1)-19.87)<1e-2
# Sample: validity parameter
f=lambda x:1/(2*(10-x)**1.5)
assert abs(f(0)-0.0158)<1e-4 and abs(f(5)-0.0447)<1e-4 and abs(f(9.9)-15.8)<0.05
assert abs(5**(2/3)-2.924)<1e-3 and abs(10-5**(2/3)-7.08)<5e-3
# Table linear well
a,ap,_,_=ai_zeros(50); Ex=np.sort(np.concatenate([-a,-ap]))
n=np.arange(1,21); Ew=(0.75*np.pi*(n-0.5))**(2/3); err=(Ew-Ex[:20])/(np.pi/(2*np.sqrt(Ew)))
tabW=[1.1155,2.3203,3.2616,4.0818,4.8263,5.5172]; tabX=[1.0188,2.3381,3.2482,4.0879,4.8201,5.5206]
tabE=[0.065,-0.017,0.015,-0.008,0.009,-0.005]
for k in range(6):
    assert abs(Ew[k]-tabW[k])<1e-4 and abs(Ex[k]-tabX[k])<1e-4
    assert abs(Ew[k]-Ex[k]-(tabW[k]-tabX[k]))<2e-4 and abs(err[k]-tabE[k])<1.5e-3
assert abs((Ew[0]-Ex[0])/Ex[0]-0.0949)<5e-4
assert abs(Ew[5]-Ex[5])/Ex[5]<1e-3 and abs(0.0034/5.5206*100-0.06)<0.005
# Sample box
L=44.883; assert abs((np.pi/(2*L))**2-1.225e-3)<1e-6
assert abs(2*L*np.sqrt(1000)/np.pi-903.57)<0.01
assert abs(2*L*np.sqrt(1000)/np.pi-903.6)<0.05
# Accuracy section: box spacings and zero spacings
E1b=(np.pi/(2*L))**2; d21=4*E1b-E1b
assert abs(d21-0.0037)<5e-5
assert abs(0.01/d21-2.7)<0.05
assert abs(np.pi*np.sqrt(1000)/L-2.2)<0.05
t1=14.134725141734695
assert abs(2*np.pi/np.log(t1/(2*np.pi))-7.75)<0.01
assert abs(2*np.pi/np.log(1000/(2*np.pi))-1.24)<0.005
# phase space
assert abs(5*np.pi-15.71)<0.01
# density box
assert abs(np.pi*np.sqrt(1000)/L-2.2135)<1e-3
# Leibniz check, linear well
Et=4.0
I=2*quad(lambda x:1/(2*np.sqrt(Et-x)),0,Et)[0]
assert abs(I-2*np.sqrt(Et))<1e-6
# Non-uniqueness: lopsided well has same Phi as |x|
El=3.0
PhiL=quad(lambda x:np.sqrt(El+2*x),-El/2,0)[0]+quad(lambda x:np.sqrt(El-2*x/3),0,1.5*El)[0]
assert abs(PhiL-4/3*El**1.5)<1e-8
# time fraction
assert abs(np.sqrt(0.1)-0.316)<1e-3
# coverage
assert abs(2*L*np.sqrt(811.1843588465)/np.pi-813.8)<0.05
# Abel numeric vs L
V0=qmnt.V0; W=1000; aa=np.sqrt(W-V0)
xab=quad(lambda u:np.log((W-u*u)/(2*np.pi)),0,aa)[0]/np.pi
assert abs(xab-44.883)<1e-3 and abs(xab-qmnt.L_of(1000))<1e-3
cf=(aa*np.log(V0/(2*np.pi*np.e**2))+np.sqrt(W)*np.log((np.sqrt(W)+aa)/(np.sqrt(W)-aa)))/np.pi
assert abs(cf-xab)<1e-8
# Problems 1-5
assert abs(2*np.pi/4-1.571)<1e-3 and abs(np.sqrt(12)-3.464)<1e-3 and abs(2*np.pi/np.sqrt(12)-1.814)<1e-3
assert int(20/np.pi)==6 and abs(20/np.pi-6.37)<0.01
assert int(np.floor(5.5))==5
Phi8=4/3*8**1.5
assert abs(Phi8-30.17)<0.01 and abs(2*Phi8-60.34)<0.01
assert abs(Phi8/np.pi-9.60)<0.01
assert abs(Phi8/np.pi+0.5-10.10)<0.01 and int(np.floor(Phi8/np.pi+0.5))==10
assert abs(Ex[9]-7.944)<1e-3 and abs(Ex[10]-8.488)<1e-3 and np.sum(Ex<=8)==10
assert abs(2*np.pi/12-0.5236)<1e-4
# Problem 7
assert abs((1.5*np.pi*0.75)**(2/3)-2.3203)<1e-4
# Problem 9 (numeric instance s=2)
s=2.0; assert abs(s**3/2*0+1.5*s**2-(s*s+s*s/2))<1e-12
# Problem 11
c=2*quad(lambda y:np.sqrt(1-y**4),0,1)[0]; assert abs(c-1.74804)<1e-5
E1=(np.pi/(2*c))**(4/3); assert abs(E1-0.8672)<1e-4
Lq=6.0; Nq=4000; h=2*Lq/(Nq+1); xg=-Lq+h*np.arange(1,Nq+1)
w=eigh_tridiagonal(2/h**2+xg**4,-np.ones(Nq-1)/h**2,select='i',select_range=(0,0))[0]
assert abs(w[0]-1.0604)<1e-3
# Problem 13
t=qmnt.zeros(1000)
assert abs(qmnt.wu_sprung_x(t[200])-22.5)<0.3 and abs(qmnt.wu_sprung_x(t[299])-28.5)<0.3
# Problem 14
v14=2*59.957*np.sqrt(1419.42248094599)/np.pi
assert abs(v14-1438.06)<0.01
# Problem 15
g=qmnt.mean_gap(np.array([200,400]))
assert abs(g[0]-1.52)<0.006 and abs(g[1]-1.34)<0.006
r=2*np.pi/np.log(2)/g
assert abs(2*np.pi/np.log(2)-9.065)<1e-3
assert abs(r[0]-6.0)<0.1 and abs(r[1]-6.8)<0.1
# Problem 21
assert abs(qmnt.Nsm(1000.0)-648.61)<0.01 and int(np.floor(qmnt.Nsm(1000.0)+0.5))==649
# Problem 23 (corrected: V=x^2, E=101, x=5)
x23=5.0; p23=abs(2*x23)/(2*(101-x23**2)**1.5)
assert abs(p23-0.00755)<1e-5 and abs(5/76**1.5-0.00755)<1e-5
print("all checks passed")
