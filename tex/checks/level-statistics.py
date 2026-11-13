import numpy as np, qmnt
from scipy.integrate import quad
from scipy.special import erfc
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
kk = np.arange(1, 501); Ek = (kk*pi/(2*L))**2
ub = 2*L*np.sqrt(Ek)/pi - 0.5
assert np.allclose(ub - (kk-0.5), 0, atol=1e-9) and np.allclose(np.diff(ub), 1, atol=1e-9)
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
sc = (t[1]-t[0])/gap(tb)
assert abs(sc - 1.128) < 2e-3
Dt = t[1]-t[0]
assert abs(Dt**2-47.43) < 0.01 and abs(tb**2-309.0) < 0.1 and abs(np.log(tb/(2*pi))-1.0288) < 2e-4
pred = -Dt**2/(24*tb**2*np.log(tb/(2*pi)))
assert abs(pred+0.0062) < 1e-4
assert abs((1.121-1.128)/1.128+0.0062) < 1e-4
ex = u[1]-u[0]
assert abs((ex-sc)/sc - pred) < 5e-4
assert abs((ex-sc)/sc + 0.0062) < 3e-4
# Poisson limit bound: |log(1-x)+x| <= x^2 for x<=1/2, and |log Q_K + s| <= (s^2+s)/K
for x in np.linspace(1e-4, 0.5, 50):
    assert abs(np.log(1-x)+x) <= x**2 + 1e-15
for K in (10, 100, 1000):
    for s in (0.5, 1.0, 2.0):
        if K >= 2*s:
            assert abs((K-1)*np.log(1-s/K)+s) <= (s**2+s)/K + 1e-12
# Sample: rare gaps
pg = lambda s: 32/pi**2*s**2*np.exp(-4*s**2/pi)
assert abs(1-np.exp(-0.1) - 0.0952) < 1e-4
Pg01 = quad(pg, 0, 0.1)[0]; assert abs(Pg01 - 1.073e-3) < 2e-6
assert abs((1-np.exp(-0.1))/Pg01 - 89) < 1
assert abs(np.exp(-2)-0.135) < 1e-3
Pg2 = quad(pg, 2, np.inf)[0]
assert abs(Pg2 - 0.017) < 1e-3
term1 = 8/pi*np.exp(-16/pi); term2 = erfc(4/np.sqrt(pi))
assert abs(term1-0.0156) < 1e-4 and abs(term2-0.0014) < 1e-4
assert abs(term1+term2-Pg2) < 1e-10
assert abs(np.exp(-2)/Pg2 - 8) < 0.5
assert abs(2*(4/pi)**2 - 32/pi**2) < 1e-12 and abs(np.sqrt(pi*4/pi)-2) < 1e-12
assert abs(32/pi**2 - 3.2423) < 1e-4
# Sample 2x2
ev = np.linalg.eigvalsh([[1,0.3],[0.3,1.2]])
assert np.allclose(ev, [0.7838,1.4162], atol=1e-4) and abs(ev[1]-ev[0]-0.6325) < 1e-4
assert abs(2*np.sqrt(0.04+0.0025)-0.4123) < 1e-4
# Jacobian determinant
assert abs(np.linalg.det([[1,0.5],[1,-0.5]])+1) < 1e-12
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
p_goe01 = 1-np.exp(-pi*0.01/4)
se1 = np.sqrt(p_goe01*(1-p_goe01)/2e5); se2 = np.sqrt(Pg01*(1-Pg01)/2e5)
assert abs(se1-2e-4) < 2e-5 and abs(se2-7e-5) < 5e-6
assert abs(np.mean(s1<0.1)-p_goe01) < 3*se1 and abs(np.mean(s2<0.1)-Pg01) < 3*se2
assert abs(np.mean(s1**2)-4/pi)<0.01 and abs(np.mean(s2**2)-3*pi/8)<0.01
# pair correlation
R2 = lambda v: 1-(np.sin(pi*v)/(pi*v))**2
assert abs(R2(0.1)-0.0325)<1e-4 and abs(pi**2*0.01/3-0.0329)<1e-4
assert abs(R2(0.5)-0.5947)<1e-4 and abs(R2(1.0)-1)<1e-12
J = quad(lambda v:(np.sin(pi*v)/(pi*v))**2, 0, 0.5)[0]
assert abs(J-0.38685)<1e-4 and abs(0.5-J-0.113)<1e-3
assert abs(pi**2/3-3.290)<1e-3
# Fig 2 reproduction claims
t10k = qmnt.zeros(10000); u10k = qmnt.Nsm(t10k)
d10 = u10k - (np.arange(1, 10001) - 0.5)
ds = np.diff(u10k)
assert abs(ds.mean() - (1 + (d10[-1]-d10[0])/9999)) < 1e-12
assert abs(d10[0]) < 1 and abs(d10[-1]) < 1
assert abs(ds.mean()-1) < 2e-4
vv = ds.var()
assert abs(vv-0.178) < abs(vv-0.273) and abs(vv-1) > 0.5
assert np.mean(ds<0.1) < 0.01
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
tz = qmnt.zeros(400)[-1]; assert abs(tz-680)<1.5        # P17
g400 = gap(tz); assert abs(g400-1.341)<1.5e-3
assert abs(gap(tz-3)-gap(tz+3)) < 6*5e-4
assert abs(2*pi/np.log(2)/g400 - 6.8) < 0.05 and 6 < 2*pi/np.log(2)/g400 < 7
assert abs(2*pi/np.log(2)-9.065)<1e-3
assert abs(32/(3*pi**2)*0.05**3-1.351e-4)<1e-7            # P20
# P13 binomial se
assert abs(np.sqrt(0.095*0.905/2e4)-0.002)<2e-4
# P5: trace invariance numerical instance
rng2 = np.random.default_rng(3)
G = rng2.normal(size=(2,2)) + 1j*rng2.normal(size=(2,2)); H = (G+G.conj().T)/2
Q, _ = np.linalg.qr(rng2.normal(size=(2,2)) + 1j*rng2.normal(size=(2,2)))
Hp = Q @ H @ Q.conj().T
assert abs(np.trace(Hp@Hp) - np.trace(H@H)) < 1e-12
# P21: MA(1) consecutive-spacing correlation
for a in (0.3, 0.6, 1.0):
    C0m, C1m, C2m = 1+a**2, -a, 0.0
    assert abs((-C0m+2*C1m-C2m)/(2*(C0m-C1m)) + (1+a)**2/(2*(1+a+a**2))) < 1e-12
assert abs(-4/6+0.667) < 1e-3
# varlink and spacing covariance, zeros 1-1000
uu = qmnt.Nsm(qmnt.zeros(1000)); dl = uu-(np.arange(1,1001)-0.5); dl = dl-dl.mean()
C0 = np.mean(dl*dl); C1 = np.mean(dl[1:]*dl[:-1]); C2 = np.mean(dl[2:]*dl[:-2])
lhs = np.var(np.diff(uu)); rhs = 2*(C0-C1)
assert abs(lhs-rhs)<0.01
assert C1 < 0 and lhs > 2*C0 and C0 < lhs/2
du = np.diff(uu); dd = du - du.mean()
covsp = np.mean(dd[:-1]*dd[1:])
assert abs(covsp - (-C0+2*C1-C2)) < 0.01
assert np.corrcoef(du[:-1], du[1:])[0,1] < 0
# P22
assert abs(quad(lambda s:s**2*np.exp(-4/pi*s**2),0,np.inf)[0]-np.sqrt(pi)/(4*(4/pi)**1.5))<1e-10
# P15: paper null
nh = np.arange(401,501); assert abs(qmnt.rms(qmnt.fluct(qmnt.zeros(500)[400:], nh))-0.263)<2e-3
print("all checks passed")
