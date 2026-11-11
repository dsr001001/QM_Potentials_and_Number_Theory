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
assert abs((Ew[0]-Ex[0])/Ew[0]-0.0967/1.1155)<1e-3 and abs(Ew[5]-Ex[5])/Ew[5]<1e-3
# Sample box
L=44.883; assert abs((np.pi/(2*L))**2-1.225e-3)<1e-6
assert abs(2*L*np.sqrt(1000)/np.pi-903.57)<0.01
# phase space
assert abs(5*np.pi-15.71)<0.01
# density box
assert abs(np.pi*np.sqrt(1000)/L-2.2135)<1e-3
# time fraction
assert abs(np.sqrt(0.1)-0.316)<1e-3
# coverage
assert abs(2*L*np.sqrt(811.1843588465)/np.pi-813.8)<0.05
# Abel numeric vs L
V0=qmnt.V0; W=1000; aa=np.sqrt(W-V0)
xab=quad(lambda u:np.log((W-u*u)/(2*np.pi)),0,aa)[0]/np.pi
assert abs(xab-44.883)<1e-3 and abs(xab-qmnt.L_of(1000))<1e-3
# closed form (Problem 16)
cf=(aa*np.log(V0/(2*np.pi*np.e**2))+np.sqrt(W)*np.log((np.sqrt(W)+aa)/(np.sqrt(W)-aa)))/np.pi
assert abs(cf-xab)<1e-8
# Problems 1-5
assert abs(2*np.pi/4-1.571)<1e-3 and abs(np.sqrt(12)-3.464)<1e-3 and abs(2*np.pi/np.sqrt(12)-1.814)<1e-3
assert int(20/np.pi)==6 and abs(20/np.pi-6.37)<0.01
assert int(np.floor(5.5))==5
assert abs(8/3*8**1.5-60.34)<0.01 and abs(4/(3*np.pi)*8**1.5-4.80)<0.01
assert abs(2*np.pi/12-0.5236)<1e-4
# Problem 7
assert abs((1.5*np.pi*0.75)**(2/3)-2.3203)<1e-4
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
v14=2*59.957*np.sqrt(1419.42248094599)/np.pi; print("P14",v14)
# Problem 15
g=qmnt.mean_gap(np.array([200,400]))
r=2*np.pi/np.log(2)/g
assert abs(2*np.pi/np.log(2)-9.065)<1e-3
assert abs(r[0]-6.0)<0.1 and abs(r[1]-6.8)<0.1
# Problem 21
assert abs(qmnt.Nsm(1000.0)-648.61)<0.01 and int(np.floor(qmnt.Nsm(1000.0)+0.5))==649
# Problem 23
assert abs(5/96**1.5-0.00532)<1e-5
print("all checks passed")
