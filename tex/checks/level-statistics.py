import numpy as np, qmnt
from scipy.integrate import quad
pi = np.pi
gap = lambda t: 2*pi/np.log(t/(2*pi))
# Sec unfolding: gaps
assert abs(gap(14.134725) - 7.750) < 2e-3
assert abs(gap(1419.42248) - 1.159) < 2e-3
# Sample: box
L = 44.883; E1 = (pi/(2*L))**2
assert abs(E1 - 1.2248e-3) < 1e-7 and abs(500**2*E1 - 306.2) < 0.1
assert abs(3*E1 - 3.674e-3) < 1e-6 and abs(1001*E1 - 1.226) < 1e-3
assert abs(1001/3 - 333.67) < 0.01
# Sample: first ten zeros, Table
t = qmnt.zeros(10); u = qmnt.Nsm(t); n = np.arange(1, 11)
tab_u = [0.449,1.570,2.393,3.671,4.317,5.593,6.565,7.294,8.771,9.348]
assert np.allclose(u, tab_u, atol=2e-3)
assert np.allclose(u-(n-0.5), [-0.051,0.070,-0.107,0.171,-0.183,0.093,0.065,-0.206,0.271,-0.152], atol=2e-3)
assert np.allclose(np.diff(u), [1.121,0.823,1.278,0.646,1.276,0.972,0.729,1.476,0.578], atol=3e-3)
assert abs(np.mean(np.diff(u)) - 0.989) < 2e-3
# local-gap shortcut
assert abs(t[1]-t[0] - 6.887) < 1e-3
tb = 0.5*(t[0]+t[1]); assert abs(tb-17.578) < 1e-3 and abs(gap(tb)-6.107) < 2e-3
assert abs((t[1]-t[0])/gap(tb) - 1.128) < 2e-3
# Sample: rare gaps
pg = lambda s: 32/pi**2*s**2*np.exp(-4*s**2/pi)
assert abs(1-np.exp(-0.1) - 0.0952) < 1e-4
Pg01 = quad(pg, 0, 0.1)[0]; assert abs(Pg01 - 1.073e-3) < 2e-6
assert abs((1-np.exp(-0.1))/Pg01 - 89) < 1
assert abs(np.exp(-2)-0.135) < 1e-3
assert abs(quad(pg, 2, np.inf)[0] - 0.017) < 1e-3
assert abs(32/pi**2 - 3.2423) < 1e-4
# Sample 2x2
ev = np.linalg.eigvalsh([[1,0.3],[0.3,1.2]])
assert np.allclose(ev, [0.7838,1.4162], atol=1e-4) and abs(ev[1]-ev[0]-0.6325) < 1e-4
assert abs(2*np.sqrt(0.04+0.0025)-0.4123) < 1e-4
# Sample shapes
assert abs(np.sqrt(2/pi)-0.798) < 1e-3 and abs(np.sqrt(pi/4)-0.886) < 1e-3
assert abs(4/pi-1-0.273) < 1e-3 and abs(3*pi/8-1-0.178) < 1e-3
assert abs(np.sqrt(3*pi/8-1)-0.422) < 1e-3
pgoe = lambda s: pi/2*s*np.exp(-pi*s**2/4)
for p in (pgoe, pg):
    assert abs(quad(p,0,np.inf)[0]-1)<1e-8 and abs(quad(lambda s:s*p(s),0,np.inf)[0]-1)<1e-8
# small-spacing
assert abs(1-np.exp(-pi*0.01/4)-0.00782) < 1e-5 and abs(pi/4*0.01-0.00785) < 1e-5
assert abs(32/(3*pi**2)-1.081) < 1e-3 and abs(32/(3*pi**2)*1e-3-1.081e-3) < 1e-6
# 2x2 simulation listing
rng = np.random.default_rng(0); x,y,z = rng.normal(size=(3,200000))
s1 = np.hypot(x,y); s1/=s1.mean(); s2 = np.sqrt(x**2+y**2+z**2); s2/=s2.mean()
assert abs(np.mean(s1<0.1)-0.0078)<1e-3 and abs(np.mean(s2<0.1)-0.0011)<1e-3
assert abs(np.mean(s1**2)-4/pi)<0.01 and abs(np.mean(s2**2)-3*pi/8)<0.01
# pair correlation
R2 = lambda v: 1-(np.sin(pi*v)/(pi*v))**2
assert abs(R2(0.1)-0.0325)<1e-4 and abs(pi**2*0.01/3-0.0329)<1e-4
assert abs(R2(0.5)-0.5947)<1e-4 and abs(R2(1.0)-1)<1e-12
J = quad(lambda v:(np.sin(pi*v)/(pi*v))**2, 0, 0.5)[0]
assert abs(J-0.38685)<1e-4 and abs(0.5-J-0.113)<1e-3
assert abs(pi**2/3-3.290)<1e-3
# Fig 2 reproduction claims
ds = np.diff(qmnt.Nsm(qmnt.zeros(10000)))
assert abs(ds.mean()-1)<1e-3
assert 0.13 < ds.var() < 0.23
assert np.mean(ds<0.1) < 0.02
# MA(1) sample and problem 9
r1 = -0.6/1.36; assert abs(r1+0.441)<1e-3 and abs(np.sqrt(2)*r1+0.624)<1e-3
assert abs(np.sqrt(2)*(-0.4)+0.566)<1e-3 and abs(-np.sqrt(2)/2+0.707)<1e-3
# acf lag1 negative (Fig acf; paper claim)
nn = np.arange(1,401); d = qmnt.fluct(qmnt.zeros(400), nn); d = d-d.mean()
assert np.mean(d[1:]*d[:-1]) < 0
# Answers
assert abs(np.exp(-1)-0.368)<1e-3                         # P2
assert abs(gap(100)-2.270)<2e-3                           # P3
assert abs(2*np.sqrt(0.29)-1.077)<1e-3                    # P7
B = 64/(9*pi); assert abs(B-2.264)<1e-3 and abs(B**3-11.60)<0.01   # P11
assert abs(2**18/(3**6*pi**3)-B**3)<1e-9
p4 = lambda s: B**3*s**4*np.exp(-B*s**2)
assert abs(quad(p4,0,np.inf)[0]-1)<1e-8 and abs(quad(lambda s:s*p4(s),0,np.inf)[0]-1)<1e-8
tz = qmnt.zeros(400)[-1]; assert abs(tz-676.1)<0.1        # P17
g400 = gap(tz); assert abs(g400-1.343)<2e-3 and 6 < 2*pi/np.log(2)/g400 < 7
assert abs(2*pi/np.log(2)-9.065)<1e-3
assert abs(32/(3*pi**2)*0.05**3-1.351e-4)<1e-7            # P20
# P13 binomial se
assert abs(np.sqrt(0.095*0.905/2e4)-0.002)<2e-4
# P21
uu = qmnt.Nsm(qmnt.zeros(1000)); dl = uu-(np.arange(1,1001)-0.5); dl = dl-dl.mean()
lhs = np.var(np.diff(uu)); rhs = 2*(np.mean(dl*dl)-np.mean(dl[1:]*dl[:-1]))
assert abs(lhs-rhs)<0.01
# P22
assert abs(quad(lambda s:s**2*np.exp(-4/pi*s**2),0,np.inf)[0]-np.sqrt(pi)/(4*(4/pi)**1.5))<1e-10
# P15: paper null
nh = np.arange(401,501); assert abs(qmnt.rms(qmnt.fluct(qmnt.zeros(500)[400:], nh))-0.263)<2e-3
print("all checks passed")
