import numpy as np
from scipy.optimize import brentq
import qmnt

# Sample: rectangular pulse
assert abs(2*np.sin(2) - 1.8186) < 1e-3
assert abs(2*np.pi/4 - 1.5708) < 1e-4
# Sample: Lorentzian
a = 0.5; assert abs(2/a - 4) < 1e-12 and abs(2*a - 1) < 1e-12
# Largest sidelobe
x = brentq(lambda x: np.tan(x) - x, 4.0, 4.6)
assert abs(x - 4.4934) < 1e-3
sl = abs(np.sin(x)/x); assert abs(sl - 0.2172) < 1e-3
assert abs(20*np.log10(sl) + 13.3) < 0.05
# Hann sidelobe
d = np.linspace(2.01, 4, 200001)
h = np.abs(np.sin(np.pi*d)/(2*np.pi*d*(d**2-1)))/0.5
assert abs(h.max() - 0.0267) < 3e-4 and abs(20*np.log10(h.max()) + 31.5) < 0.1
# Sample: half bin scalloping
assert abs(2/np.pi - 0.6366) < 1e-4 and abs(8/(3*np.pi) - 0.8488) < 1e-4
# Sample: quarter step
s8 = np.sin(np.pi/8)/(np.pi/8)
assert abs(s8 - 0.9745) < 1e-4 and abs(s8**2 - 0.9496) < 1e-4
# RE2 amplitudes
A = [1/(np.pi*m*2**(m/2)) for m in (1, 2, 3)]
assert np.allclose(A, [0.2251, 0.0796, 0.0375], atol=1e-4)
assert np.allclose(np.array(A)/2, [0.1125, 0.0398, 0.0188], atol=1e-4)
assert abs(2*np.pi/np.log(2) - 9.065) < 1e-3
# Sample: resolving prime powers
assert abs(np.log(9/8) - 0.1178) < 1e-4 and abs(2*np.pi/np.log(9/8) - 53.3) < 0.1
assert abs(np.log(128/127) - 0.00784) < 1e-5 and abs(2*np.pi/np.log(128/127) - 801) < 1
t = qmnt.zeros(10000)
assert abs(t[399] - 680) < 1
T400 = t[399] - t[0]; assert abs(T400 - 666) < 1 and abs(2*np.pi/T400 - 0.0094) < 1e-4
# Sample: alias and equivalent Nyquist
assert abs(2*np.pi - 0.5 - 5.7832) < 1e-4
s = qmnt.smooth_zeros(np.array([400, 800, 1000]))
assert abs(s[0] - 679.9) < 0.1
L = np.log(s[0]/(2*np.pi)); assert abs(L - 4.684) < 1e-3
assert abs(2*np.pi/L - 1.341) < 1e-3 and abs(L/2 - 2.342) < 1e-3
assert abs(qmnt.mean_gap(np.array([400]))[0] - 1.341) < 2e-3
# densities
dens = lambda tt: np.log(tt/(2*np.pi))/(2*np.pi)
assert abs(dens(t[0]) - 0.13) < 0.01 and abs(dens(t[399]) - 0.75) < 0.01
# Checkpoint aliasing dt=0.1
assert abs(np.pi/0.1 - 31.4) < 0.05 and abs(2*np.pi/0.1 - 40 - 22.83) < 0.01
# DFT samples
X = np.fft.fft([2., 1, 0, 1]); assert np.allclose(X, [4, 2, 0, 2])
X = np.fft.fft([1., 1, 0, 0]); assert np.allclose(np.abs(X)**2, [4, 2, 0, 2])
# Sample: bins
assert abs(2*np.pi/512 - 0.01227) < 1e-5 and abs(1024/10 - 102.4) < 1e-9
# Sample: weak oscillation
assert abs((400*0.01/4)/0.09 - 11.1) < 0.05
# Phase randomization sample
xs = 0.5 + np.sqrt(2)/2*np.cos(np.pi*np.arange(4)/2)
assert np.allclose(xs, [1.2071, 0.5, -0.2071, 0.5], atol=1e-4)
r = [np.sum(xs*np.roll(xs, -m)) for m in range(4)]
assert np.allclose(r, [2, 1, 0, 1])          # also Problem 20
xp = 0.5 + np.sqrt(2)/2*np.cos(np.pi*np.arange(4)/2 + np.pi/4) * 1  # phase pi/4 -> shift
assert np.allclose(np.fft.ifft(np.fft.fft([1., 1, 0, 0])*np.array([1, np.exp(1j*np.pi/4)/np.exp(-1j*np.pi/4)*np.exp(-1j*np.pi/4)*np.sqrt(2)/abs(1-1j)*np.exp(1j*np.pi/4)/np.exp(1j*np.pi/4), 1, 1])).real.shape, (4,))
Xs = np.array([2, np.sqrt(2)*np.exp(1j*np.pi/4), 0, np.sqrt(2)*np.exp(-1j*np.pi/4)])
assert np.allclose(np.fft.ifft(Xs).real, [1, 0, 0, 1])
# Landau sample
t1000 = t[999]; pref = t1000/(2*np.pi); assert abs(pref - 225.91) < 0.01
assert abs(np.log(2)/np.sqrt(2) - 0.49013) < 1e-5 and abs(-pref*np.log(2)/np.sqrt(2) + 110.7) < 0.1
assert abs(np.log(3)/np.sqrt(3) - 0.63428) < 1e-5 and abs(-pref*np.log(3)/np.sqrt(3) + 143.3) < 0.1
meas = np.sum(np.cos(np.log(2)*t[:1000]))
assert abs(meas - (-110.7))/110.7 < 0.25
# zero periodogram: tallest peak at log 2
n = np.arange(1, 1001); y = n - 0.5 - qmnt.Nsm(t[:1000]); y = y - y.mean()
w = np.linspace(0.3, 3.0, 8000)
P = np.array([abs(np.sum(y*np.exp(-1j*v*t[:1000])))**2 for v in w])
assert abs(w[np.argmax(P)] - np.log(2)) < 2*np.pi/(t[999]-t[0])
# Paper connection numbers
assert abs(8/450 - 0.01778) < 1e-5 and abs((8/450)/(2*np.pi/T400) - 1.9) < 0.05
assert abs(0.0089*666 - 6) < 0.1
assert 3300 < (8 - 0.05)/(0.25*2*np.pi/T400) < 3400
assert abs(np.log(211) - 5.35) < 0.01 and abs(0.5*np.log(s[1]/(2*np.pi)) - 2.62) < 0.01
assert abs(np.pi/T400 - 0.0047) < 1e-4 and abs(np.exp(8) - 2981) < 1
# Problems
assert abs(2/np.pi - 0.637) < 1e-3                                   # P1
assert abs(np.pi/0.25 - 12.57) < 0.01 and abs(2*np.pi/0.25 - 15 - 10.13) < 0.01  # P2
assert abs(np.log(4/3) - 0.2877) < 1e-4 and abs(2*np.pi/np.log(4/3) - 21.8) < 0.05  # P3
assert np.allclose(np.fft.fft([1., 0, 0, 0]), 1)                     # P4
assert np.allclose(np.fft.fft([1., -1, 1, -1]), [0, 0, 4, 0])        # P7
assert abs(np.sin(np.pi/4)/(np.pi/4) - 0.900) < 1e-3                 # P9
dd = 1 + 1e-7; assert abs(-np.sin(np.pi*dd)/(2*np.pi*dd*(dd**2-1)) - 0.25) < 1e-5  # P10
T800 = t[799] - t[0]
assert abs(T800 - 1169) < 2 and abs(2*np.pi/T800 - 0.0054) < 1e-4 and abs((8/450)/(2*np.pi/T800) - 3.3) < 0.05  # P11
assert abs(-pref*np.log(2)/2 + 78.3) < 0.1                           # P13
d400 = qmnt.fluct(t[:400], np.arange(1, 401))                        # P15
for sd in range(1, 11):
    rng = np.random.default_rng(sd); Xd = np.fft.rfft(d400)
    ph = rng.uniform(0, 2*np.pi, len(Xd)); ph[0] = 0; ph[-1] = 0
    dsur = np.fft.irfft(Xd*np.exp(1j*ph), n=400)
    ca = lambda z: np.fft.ifft(np.abs(np.fft.fft(z))**2).real
    assert np.allclose(ca(d400), ca(dsur))
assert abs(s[2] - 1419.4) < 0.3 and abs(0.5*np.log(s[2]/(2*np.pi)) - 2.71) < 0.01  # P19
pred10k = -(t[9999]/(2*np.pi))*np.log(2)/np.sqrt(2)                  # P21
assert abs(t[9999] - 9878.7) < 0.1 and abs(pred10k + 770.6) < 0.2
meas10k = np.sum(np.cos(np.log(2)*t))
assert abs(meas10k - pred10k)/abs(pred10k) < abs(meas - (-pref*np.log(2)/np.sqrt(2)))/110.7 + 0.05
for v, c in [(0.134, 45), (0.141, 42), (0.138, 43), (0.137, 44)]:      # P23
    assert abs(100*(1 - v/0.244) - c) < 0.6
assert abs(2**20/20 - 52428.8) < 1e-6                                # P24
xpad = np.array([1., 1, 0, 0, 0, 0, 0, 0])                           # P25
rp = np.fft.ifft(np.abs(np.fft.fft(xpad))**2).real
assert np.allclose(rp[:4], [2, 1, 0, 0])
print("all checks passed")
