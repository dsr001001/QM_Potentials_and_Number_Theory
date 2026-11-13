import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad

def box(N, L, k):
    h = 2*L/(N+1)
    return 4/h**2*np.sin(k*np.pi*h/(4*L))**2

# Sample: second difference of sin
for h, err in [(0.1, 7.010e-4), (0.05, 1.753e-4)]:
    d = (np.sin(1+h)-2*np.sin(1)+np.sin(1-h))/h**2
    assert abs((d+np.sin(1)) - err) < 2e-6
e1 = (np.sin(1.1)-2*np.sin(1)+np.sin(0.9))/0.01+np.sin(1)
e2 = (np.sin(1.05)-2*np.sin(1)+np.sin(0.95))/0.0025+np.sin(1)
assert abs(e1/e2-4.00) < 0.01
assert abs(0.01/12*np.sin(1)-7.012e-4) < 1e-6

# Sample: three-point box
H3 = np.array([[8,-4,0],[-4,8,-4],[0,-4,8.]])
ev = np.linalg.eigvalsh(H3)
assert np.allclose(ev, [2.34315, 8, 13.65685], atol=1e-5)
ex = [(np.pi/2)**2, np.pi**2, (1.5*np.pi)**2]
rel = [(a-b)/a for a, b in zip(ex, ev)]
assert np.allclose(rel, [0.0504, 0.189, 0.385], atol=6e-4)

# Sample: eigenvalue nudged by hand (Eq. hf)
w, U = np.linalg.eigh(H3)
u = U[:, 0]
assert np.allclose(np.abs(u), [0.5, np.sqrt(0.5), 0.5], atol=1e-12)
assert abs(u[1]**2-0.5) < 1e-12
H3e = H3.copy(); H3e[1, 1] += 0.01
lo = np.linalg.eigvalsh(H3e)[0]
assert abs(lo-2.3481435) < 1e-7
assert abs((lo-w[0])-4.998e-3) < 1e-6
assert abs(w[0]+0.005-2.348146) < 1e-6

# Sample: normalizing the three-point ground state
psi3 = np.abs(u)/np.sqrt(0.5)
assert np.allclose(psi3, [0.7071, 1, 0.7071], atol=1e-4)
xj = np.array([-0.5, 0, 0.5])
assert abs(np.sum(u**2*np.cos(np.pi*xj))-0.5) < 1e-12
assert abs(quad(lambda x: np.cos(np.pi*x/2)**2*np.cos(np.pi*x), -1, 1)[0]-0.5) < 1e-10
assert abs(quad(lambda x: np.cos(np.pi*x/2)**2, -1, 1)[0]-1) < 1e-10

# Table ppw: displayed value, half a unit in the last displayed digit
for n, v, tol in [(4,.189,5e-4),(6,.0881,5e-5),(8,.0504,5e-5),(12,.0226,5e-5),(24,.00570,5e-6),(50,.0013,5e-5)]:
    assert abs(1-(np.sin(np.pi/n)/(np.pi/n))**2 - v) < tol

# Sample RE3
L, N = 44.883, 6000
h = 2*L/(N+1)
assert abs(h-0.0149585) < 1e-7
assert abs(h**2-2.23757e-4) < 1e-9 and abs(4/h**2-17876.5) < 0.1
E = (500*np.pi/(2*L))**2; Eh = box(N, L, 500)
assert abs(E-306.21) < 0.01 and abs(Eh-304.46) < 0.01
assert abs(E-Eh-1.744) < 0.002 and abs(h**2*E**2/12-1.748) < 0.002
assert abs((E-Eh)/E-0.00570) < 1e-5
spc = 1001*(np.pi/(2*L))**2
assert abs(spc-1.226) < 1e-3 and abs((E-Eh)/spc-1.42) < 0.01
assert abs(2*(N+1)/500-24.0) < 0.01
E1box = (np.pi/(2*L))**2
assert abs(E1box-1.22483e-3) < 1e-8
assert abs(h**2*E1box**2/12-3e-11) < 0.5e-11
assert abs(np.sqrt(h)-0.12) < 0.005   # rescaling factor sqrt(h)

# Harmonic oscillator sample
Nh = 199; hh = 20/200; x = -10+hh*np.arange(1, Nh+1)
e0 = eigh_tridiagonal(2/hh**2+x**2, -np.ones(Nh-1)/hh**2, eigvals_only=True,
                      select='i', select_range=(0, 0))[0]
assert abs(e0-0.999375) < 5e-6        # 'within a few times 1e-6'
assert abs(0.01/12-8.3e-4) < 1e-5     # box estimate in HO sample

# WS numbers in text
V0 = 3.10073*np.pi
assert abs(V0-9.7) < 0.05
assert abs(np.sqrt(811.18-V0)-28.31) < 0.01
assert abs(2*np.pi/(np.sqrt(811.18-V0)*h)-14.8) < 0.1
assert abs(2*np.pi/np.log(811.18/(2*np.pi))-1.293) < 0.002
assert abs(2/0.263-7.6) < 0.05   # 'about eight times'
assert abs((np.pi/np.log(V0/(2*np.pi)))**2-51.3) < 0.2
assert abs(V0/(2*np.pi)-1.550) < 5e-4 and np.log(V0/(2*np.pi)) > 0
def x_of_V(V):
    s, r = np.sqrt(V-V0), np.sqrt(V)
    return (s*np.log(V0/(2*np.pi*np.e**2))+r*np.log((r+s)**2/V0))/np.pi
assert abs(x_of_V(1000.)-44.883) < 2e-3
eta = 1e-4   # slope of x(eta) at the bottom of the well
assert abs(x_of_V(V0+eta**2)/eta-np.log(V0/(2*np.pi))/np.pi) < 1e-6
assert abs(2*0.03*2-0.12) < 1e-12     # order-of-magnitude estimate

# Richardson sample
Ec, Ef = box(3, 1, 1), box(7, 1, 1); Er = (4*Ef-Ec)/3; E1 = np.pi**2/4
assert abs(Ec-2.343146) < 1e-6 and abs(Ef-2.435855) < 1e-6 and abs(Er-2.466758) < 1e-6
assert abs(E1-Er-6.4e-4) < 1e-5 and abs(0.0625*E1**3/1440-6.5e-4) < 1e-5

# Brute force vs Richardson
hb = np.sqrt(12*0.01226/E**2); assert abs(2*L/hb-71700) < 300
assert abs(h**4-5.007e-8) < 1e-11 and abs(E**3-2.871e7) < 1e4
Er500 = (4*box(12001, L, 500)-box(6000, L, 500))/3
assert abs((E-Er500)/spc-8e-4) < 1.5e-4
assert abs(h**4*E**3/1440-1.0e-3) < 1e-4

# leakage sample, problem 19
assert abs((811.1843588-2.5-808.5)-(811.1843588-(808.5+2.5))) < 1e-12
assert abs((811.1843588-2.5-808.5)-0.1843588) < 1e-9
assert abs((50-0.3-49.65)-0.05) < 1e-12

# Problems
assert abs(119.914/8001-0.0149874) < 1e-7                     # P1
assert abs(17998/6000**2-5.0e-4) < 1e-6                         # P2
assert abs((np.exp(0.01)-1)/0.01-1.0050167) < 1e-7              # P3
assert abs((4*2.343146-2)/3-2.457528) < 1e-6                     # P7
assert abs(E1-2.457528-0.00987) < 1e-4
r = 24001/12001; assert abs((4/r**2-1)/3-2.78e-5) < 2e-7          # P11
assert abs(r-1.99992) < 1e-5
lv = eigh_tridiagonal(2/h**2*np.ones(N), -np.ones(N-1)/h**2, eigvals_only=True,
        select='i', select_range=(499, 499))[0]
assert abs(lv-Eh) < 1e-8                                           # P15
assert abs(4/h**2*np.cos(np.pi/(2*(N+1)))**2-17876.5) < 0.1       # P17
assert abs((np.pi/2)**2-2.5) < 0.05                               # P17 'about 2.5 times'
d = (np.sin(1.1)-np.sin(0.9))/0.2                                 # P21
assert abs(d-0.5394022) < 1e-7 and abs(d-np.cos(1)+9.0005e-4) < 1e-8
b = h**2*811.18**2/12; assert abs(b-12.3) < 0.05                  # P23
assert abs(b/1.293-9.5) < 0.05
errs = [E1-box(n-1, 1, 1) for n in [4, 8, 16, 32]]                # P25
assert np.allclose(errs, [0.12426, 0.031546, 0.0079170, 0.0019811], rtol=1e-3)
rat = [errs[i]/errs[i+1] for i in range(3)]
assert np.allclose(rat, [3.94, 3.98, 4.00], atol=0.006)
assert abs((4*Ec-Ef)/3-2.312243) < 1e-6                           # P31
assert abs(E1-(4*Ec-Ef)/3-0.155) < 1e-3
print('all checks passed')
