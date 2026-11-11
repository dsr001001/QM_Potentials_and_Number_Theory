"""Chapter-opener artwork and cover for "Potentials, Primes and Prediction".

Every picture is computed from the mathematics of its chapter (no stock images).
Run from tex/figs/art:  python make_art.py [slug ...]
Writes <slug>.png (banner 6.4 x 2.4 in) and cover.png (A4 portrait).
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "code"))
import qmnt

BG = "#0a0f1e"
W, H = 6.4, 2.4
DPI = 220
rng = np.random.default_rng(7)


def canvas(w=W, h=H):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_axis_off()
    return fig, ax


def save(fig, name):
    fig.savefig(f"{name}.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def glow(ax, x, y, color, lw=1.2, alpha=1.0):
    for k, a in ((6, 0.04), (3.5, 0.08), (2, 0.18)):
        ax.plot(x, y, color=color, lw=lw * k, alpha=a * alpha, solid_capstyle="round")
    ax.plot(x, y, color=color, lw=lw, alpha=alpha)


def hsv_phase(Z):
    """Domain colouring: hue = argument, brightness from modulus."""
    h = (np.angle(Z) / (2 * np.pi)) % 1.0
    m = np.log1p(np.abs(Z))
    v = 0.55 + 0.45 * (m % 1.0) ** 0.6
    s = np.full_like(h, 0.85)
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(np.dstack([h, s, v]))


ART = {}
CAPTION = {}


def art(slug, caption):
    def deco(f):
        ART[slug] = f
        CAPTION[slug] = caption
        return f
    return deco


# ------------------------------------------------------------------ Part I
@art("intro", "The heights of the first 60 Riemann zeros, drawn as energy levels; colour shows how far each sits from its smooth position.")
def _():
    fig, ax = canvas()
    t = qmnt.zeros(60); n = np.arange(1, 61); f = qmnt.fluct(t, n)
    s = np.linspace(10, t[-1] + 3, 3000)
    for p, col in ((2, "#7fd4ff"), (3, "#ffd27f"), (5, "#b48cff")):
        glow(ax, s, 0.5 + 0.28 * np.sin(s * np.log(p)) / np.sqrt(p), col, lw=0.6, alpha=0.45)
    for ti, fi in zip(t, f):
        c = plt.cm.plasma(0.55 + 0.4 * np.tanh(1.5 * fi))
        for lw, a in ((7, 0.05), (3, 0.15), (1.0, 1.0)):
            ax.plot([ti, ti], [0.12, 0.88], color=c, lw=lw, alpha=a)
    ax.set_xlim(10, t[-1] + 3); ax.set_ylim(0, 1)
    save(fig, "intro")

@art("complex", "Domain colouring of $(z^5-1)/(z^2+1)$: hue is the argument, the rings are levels of the modulus.")
def _():
    fig, ax = canvas()
    x = np.linspace(-3.2, 3.2, 1400); y = np.linspace(-1.2, 1.2, 525)
    X, Y = np.meshgrid(x, y); Z = X + 1j * Y
    ax.imshow(hsv_phase((Z ** 5 - 1) / (Z ** 2 + 1)), extent=[-3.2, 3.2, -1.2, 1.2], origin="lower", aspect="auto")
    save(fig, "complex")


@art("series", "Partial sums of the alternating series $\\sum(-1)^{k+1}k^{-s}$ at the first zero, $s=\\tfrac12+14.1347\\ldots i$, spiral into $0$ (the gold dot).")
def _():
    fig, ax = canvas()
    s = 0.5 + 1j * float(qmnt.zeros(1)[0]); k = np.arange(1, 20001)
    z = np.cumsum((-1.0) ** (k + 1) * k ** (-s)); z = 0.5 * (z[1:] + z[:-1])
    z = z[3:]
    pts = np.c_[z.real, z.imag]; seg = np.stack([pts[:-1], pts[1:]], 1)
    for lw, a in ((5, 0.06), (0.9, 0.95)):
        lc = LineCollection(seg, cmap="cool", lw=lw, alpha=a); lc.set_array(np.log(np.arange(len(seg)) + 1)); ax.add_collection(lc)
    ax.scatter([0], [0], s=25, color="#ffb347", zorder=5)
    r = np.abs(z[40:]).max() * 1.1
    ax.set_xlim(-r * W / H, r * W / H); ax.set_ylim(-r, r)
    save(fig, "series")

@art("asymptotics", "Newton's method for $z^3=1$: each starting point is coloured by the root it reaches; the boundaries are fractal.")
def _():
    fig, ax = canvas()
    x = np.linspace(-2.4, 2.4, 1300); y = np.linspace(-0.9, 0.9, 490)
    X, Y = np.meshgrid(x, y); Z = X + 1j * Y; it = np.zeros(Z.shape)
    for i in range(40):
        Zn = Z - (Z ** 3 - 1) / (3 * Z ** 2 + 1e-30)
        it += (np.abs(Zn - Z) > 1e-6)
        Z = Zn
    roots = np.exp(2j * np.pi * np.arange(3) / 3)
    which = np.argmin(np.abs(Z[..., None] - roots), -1)
    cols = np.array([[0.98, 0.45, 0.35], [0.30, 0.75, 0.95], [0.75, 0.45, 0.95]])
    img = cols[which] * (0.25 + 0.75 * np.exp(-it / 10))[..., None]
    ax.imshow(img, extent=[-2.4, 2.4, -0.9, 0.9], origin="lower", aspect="auto")
    save(fig, "asymptotics")

@art("fourier-series", "Fourier partial sums of a square wave with 1, 3, 5, ..., 39 terms; the overshoot near each jump never disappears (Gibbs).")
def _():
    fig, ax = canvas()
    x = np.linspace(0, 4 * np.pi, 3000)
    cm = plt.cm.viridis
    for j, K in enumerate(range(1, 40, 2)):
        y = sum(4 / np.pi * np.sin(k * x) / k for k in range(1, K + 1, 2))
        ax.plot(x, y, color=cm(j / 20), lw=0.7, alpha=0.75)
    glow(ax, x, np.sign(np.sin(x)), "#ffffff", lw=0.6, alpha=0.6)
    ax.set_xlim(0, 4 * np.pi); ax.set_ylim(-1.5, 1.5)
    save(fig, "fourier-series")


@art("fourier-transform", "A spectrogram: a signal whose frequency glides upward, with two harmonics, seen through short sliding windows.")
def _():
    fig, ax = canvas()
    fs = 2000; t = np.arange(0, 6, 1 / fs)
    ph = 2 * np.pi * (40 * t + 18 * t ** 2)
    sig = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph) + 0.15 * rng.normal(size=t.size)
    ax.specgram(sig, NFFT=256, Fs=fs, noverlap=230, cmap="magma", vmin=-60)
    ax.set_ylim(0, 1000); ax.set_xlim(0, 6)
    save(fig, "fourier-transform")


@art("linear-algebra", "Flow lines of $\\dot{\\mathbf x}=A\\mathbf x$ for a $2\\times2$ matrix $A$ with complex eigenvalues: every path spirals inward.")
def _():
    fig, ax = canvas()
    A = np.array([[-0.15, 1.0], [-0.55, -0.12]])
    x = np.linspace(-3.2, 3.2, 300); y = np.linspace(-1.2, 1.2, 120)
    X, Y = np.meshgrid(x, y); U = A[0, 0] * X + A[0, 1] * Y; Vv = A[1, 0] * X + A[1, 1] * Y
    sp = np.hypot(U, Vv)
    ax.streamplot(X, Y, U, Vv, color=sp, cmap="cool", density=2.4, linewidth=0.7, arrowsize=0.6)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.2, 1.2)
    save(fig, "linear-algebra")

@art("eigen", "Chladni figures: nodal lines of eigenmodes of a vibrating square plate, the 2D cousins of the string's standing waves.")
def _():
    fig, ax = canvas()
    x = np.linspace(0, 1, 400); X, Y = np.meshgrid(x, x)
    modes = [(1, 4), (2, 5), (3, 7), (2, 9), (5, 8), (4, 11)]
    for i, (a, b) in enumerate(modes):
        F = np.cos(a * np.pi * X) * np.cos(b * np.pi * Y) - np.cos(b * np.pi * X) * np.cos(a * np.pi * Y)
        ext = [i * 1.07, i * 1.07 + 1, 0, 1]
        ax.imshow(np.abs(F) ** 0.35, extent=ext, cmap="bone_r", alpha=0.25, origin="lower")
        ax.contour(np.linspace(ext[0], ext[1], 400), x, F, [0], colors=[plt.cm.spring(i / 6)], linewidths=0.9)
    ax.set_xlim(-0.05, 6 * 1.07); ax.set_ylim(-0.08, 1.08)
    save(fig, "eigen")


@art("least-squares", "Polynomials of degree 1 to 12 fitted by least squares to the same 13 noisy points: more freedom, wilder curves.")
def _():
    fig, ax = canvas()
    xs = np.linspace(0, 1, 13); ys = np.sin(2 * np.pi * xs) + 0.25 * rng.normal(size=13)
    xx = np.linspace(-0.02, 1.02, 800)
    for d in range(1, 13):
        c = np.polyfit(xs, ys, d)
        ax.plot(xx, np.polyval(c, xx), color=plt.cm.plasma(d / 13), lw=0.9, alpha=0.8)
    ax.scatter(xs, ys, s=18, color="white", zorder=5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-2.2, 2.2)
    save(fig, "least-squares")


# ------------------------------------------------------------------ Part II
@art("schrodinger", "A quantum carpet: $|\\psi(x,t)|^2$ for a wave packet bouncing in a box (time runs left to right); the pattern is pure interference.")
def _():
    fig, ax = canvas()
    x = np.linspace(0, 1, 500); t = np.linspace(0, 1 / np.pi, 1300)
    n = np.arange(1, 70)
    c = np.sin(n * np.pi * 0.5) * np.exp(-(n / 28.0) ** 2)
    S = np.sin(np.pi * np.outer(n, x))
    psi = (np.exp(-1j * np.outer(t, n ** 2) * np.pi ** 2) * c) @ S
    P = np.abs(psi) ** 2; P = P / P.max()
    ax.imshow(P.T ** 0.4, aspect="auto", cmap="inferno", origin="lower")
    save(fig, "schrodinger")

@art("bound-states", "The first eight bound states of an anharmonic well, each drawn at its energy.")
def _():
    fig, ax = canvas()
    L = 4.0; N = 1200; x = qmnt.grid(L, N); V = 0.6 * x ** 4 / 4 + 0.5 * x ** 2
    E, psi = qmnt.fd_levels(V, L, 8, return_vectors=True)
    ax.fill_between(x, V, 60, color="#1b2440")
    ax.plot(x, V, color="#8899cc", lw=1.2)
    for k in range(8):
        y = E[k] + 2.2 * psi[:, k] * (1 if psi[N // 2 + 40, k] >= 0 else -1)
        glow(ax, x, y, plt.cm.spring(k / 8), lw=0.9)
        ax.plot(x, np.full_like(x, E[k]), color="white", lw=0.3, alpha=0.3)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-1, E[7] + 3)
    save(fig, "bound-states")


@art("wkb", "Quantized orbits in phase space: the curves $p^2+V(x)=E_n$ for the Wu--Sprung potential, one for each of the first 30 levels.")
def _():
    fig, ax = canvas()
    x = np.linspace(-30, 30, 6000); V = qmnt.wu_sprung_V(x)
    s = qmnt.smooth_zeros(np.arange(1, 31))
    for k, E in enumerate(s):
        m = E - V >= 0; p = np.sqrt(np.clip(E - V, 0, None))
        xx = np.r_[x[m], x[m][::-1]]; pp = np.r_[p[m], -p[m][::-1]]
        ax.plot(xx, pp, color=plt.cm.cool(k / 30), lw=0.8)
    xm = float(qmnt.wu_sprung_x(s[-1])) * 1.05
    ax.set_xlim(-xm * W / H * 0.42, xm * W / H * 0.42); ax.set_ylim(-1.08 * np.sqrt(s[-1] - qmnt.V0), 1.08 * np.sqrt(s[-1] - qmnt.V0))
    save(fig, "wkb")

@art("perturbation", "Eigenvalues of $\\cos\\lambda\\,A+\\sin\\lambda\\,B$ for two random symmetric matrices, as $\\lambda$ runs from $0$ to $\\pi$: levels approach, then swerve apart, never crossing.")
def _():
    fig, ax = canvas()
    n = 40
    G1 = rng.normal(size=(n, n)); G2 = rng.normal(size=(n, n))
    A1, A2 = (G1 + G1.T) / 2, (G2 + G2.T) / 2
    lam = np.linspace(0, np.pi, 700)
    ev = np.array([np.linalg.eigvalsh(np.cos(l) * A1 + np.sin(l) * A2) for l in lam])
    for k in range(n):
        glow(ax, lam, ev[:, k], plt.cm.viridis(0.15 + 0.85 * k / n), lw=0.6, alpha=0.9)
    ax.set_xlim(0, np.pi); ax.set_ylim(-3.2, 3.2)
    save(fig, "perturbation")

@art("non-hermitian", "Real parts of the eigenvalues of a PT-symmetric chain (gain on one half, loss on the other) as the strength grows left to right: pairs of real levels meet and turn complex (brighter = larger imaginary part).")
def _():
    fig, ax = canvas()
    n = 16; off = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    gam = np.linspace(0, 1.4, 1400); pot = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    for g in gam:
        ev = np.linalg.eigvals(off + 1j * g * np.diag(pot))
        ax.scatter(np.full(n, g), ev.real, s=0.5, c=np.abs(ev.imag), cmap="plasma", vmin=0, vmax=0.9, alpha=0.9)
    ax.set_xlim(0, 1.4); ax.set_ylim(-2.15, 2.15)
    save(fig, "non-hermitian")

# ------------------------------------------------------------------ Part III
@art("primes", "The Sacks spiral: the integer $n$ sits at radius $\\sqrt n$ and angle $2\\pi\\sqrt n$; primes are lit. Curves of primes appear.")
def _():
    fig, ax = canvas()
    N = 40000; n = np.arange(1, N + 1); r = np.sqrt(n); th = 2 * np.pi * r
    x, y = r * np.cos(th), r * np.sin(th)
    isp = np.zeros(N + 1, bool); isp[qmnt.primes_upto(N)] = True; isp = isp[1:]
    ax.scatter(x[~isp], y[~isp], s=0.08, color="#223355")
    ax.scatter(x[isp], y[isp], s=0.35, c=r[isp], cmap="autumn")
    ax.set_xlim(-205 * W / H / 2.6, 205 * W / H / 2.6); ax.set_ylim(-80, 80)
    save(fig, "primes")


@art("zeta", "Domain colouring of $\\zeta(s)$ for $-1.5\\le\\sigma\\le2.5$, $0\\le t\\le 45$ (turned on its side): every point where all colours meet is a zero.")
def _():
    import mpmath
    fig, ax = canvas()
    ts = np.linspace(0, 45, 900); ss = np.linspace(-1.5, 2.5, 330)
    Z = np.array([[complex(mpmath.zeta(complex(s, t))) if (s, t) != (1, 0) else 1e9 for t in ts] for s in ss])
    ax.imshow(hsv_phase(Z), extent=[0, 45, -1.5, 2.5], origin="lower", aspect="auto")
    ax.axhline(0.5, color="white", lw=0.5, alpha=0.5)
    save(fig, "zeta")


@art("counting-zeros", "The staircase $N(t)$ counting zeros up to height 140, and the smooth curve $N_{\\mathrm{sm}}(t)$ it hugs.")
def _():
    fig, ax = canvas()
    z = qmnt.zeros(80); t = np.linspace(10, 140, 4000)
    N = np.searchsorted(z, t)
    ax.fill_between(t, 0, N, color="#2a3d6e", alpha=0.5, step="pre")
    glow(ax, t, N, "#7fd4ff", lw=1.0)
    glow(ax, t, qmnt.Nsm(t), "#ffb347", lw=0.9)
    ax.set_xlim(10, 140); ax.set_ylim(-2, 36)
    save(fig, "counting-zeros")


@art("explicit-formula", "One prime wave each for $p=2,3,5,\\dots,29$, and below them their sum, which tracks the true fluctuation of the zero count.")
def _():
    fig, ax = canvas()
    t = np.linspace(14, 120, 4000); P = qmnt.primes_upto(29)
    for i, p in enumerate(P):
        y = 2.2 + 0.32 * i - np.sin(t * np.log(p)) / np.sqrt(p) * 0.35
        ax.plot(t, y, color=plt.cm.cool(i / len(P)), lw=0.8, alpha=0.85)
    z = qmnt.zeros(60); S = np.searchsorted(z, t) + 0.5 - qmnt.Nsm(t) - 0.5
    ax.plot(t, 0.8 + 0.7 * S, color="#ffffff", lw=0.6, alpha=0.5, drawstyle="steps-post")
    glow(ax, t, 0.8 + 0.7 * qmnt.Nfl(t, P), "#ffb347", lw=1.0)
    ax.set_xlim(14, 120); ax.set_ylim(-0.2, 5.8)
    save(fig, "explicit-formula")


# ------------------------------------------------------------------ Part IV
@art("probability", "Two hundred random walks of 1000 steps fanning out from the same start: their spread grows like the square root of time.")
def _():
    fig, ax = canvas()
    steps = rng.choice([-1, 1], size=(200, 1000)).cumsum(1)
    t = np.arange(1000)
    for i in range(200):
        ax.plot(t, steps[i], color=plt.cm.cool(0.5 + steps[i, -1] / 120), lw=0.4, alpha=0.4)
    glow(ax, t, 2 * np.sqrt(t), "#ffb347", lw=0.8); glow(ax, t, -2 * np.sqrt(t), "#ffb347", lw=0.8)
    ax.set_xlim(0, 1000); ax.set_ylim(-90, 90)
    save(fig, "probability")


@art("level-statistics", "Three barcodes with the same mean spacing: independent random levels (top), Riemann zeros 1000 to 1100 (middle), a perfect lattice (bottom).")
def _():
    fig, ax = canvas()
    n = np.arange(1000, 1101); z = qmnt.zeros(1101)[999:1101]
    u = qmnt.Nsm(z); u = u - u[0]
    pois = np.cumsum(rng.exponential(size=101)); pois = pois - pois[0]
    for row, (vals, col) in enumerate(((np.arange(101.0), "#9ad0ff"), (u, "#ffb347"), (pois, "#ff6fa8"))):
        for v in vals:
            ax.plot([v, v], [0.08 + row / 3, 0.30 + row / 3], color=col, lw=1.1)
    ax.set_xlim(-1, 101); ax.set_ylim(0, 1)
    save(fig, "level-statistics")


@art("hilbert-polya", "The flow of the Berry--Keating Hamiltonian $H=xp$ in phase space: hyperbolas $xp=E$, the classical orbits behind a conjectured zeta operator.")
def _():
    fig, ax = canvas()
    x = np.linspace(0.02, 9, 2000)
    for k, E in enumerate(np.linspace(0.3, 12, 26)):
        col = plt.cm.cool(k / 26)
        for sx in (1, -1):
            for sp in (1, -1):
                ax.plot(sx * x, sp * E / x, color=col, lw=0.7, alpha=0.8)
    ax.set_xlim(-8, 8); ax.set_ylim(-3, 3)
    save(fig, "hilbert-polya")


@art("wu-sprung", "The Wu--Sprung potential, built so that its levels follow the smooth zeros; the first 60 levels are drawn across the well.")
def _():
    fig, ax = canvas()
    x = np.linspace(-44.8, 44.8, 2000); V = qmnt.wu_sprung_V(x)
    s = qmnt.smooth_zeros(np.arange(1, 61))
    ax.fill_between(x, V, 400, color="#1b2440")
    for k, E in enumerate(s):
        xe = float(qmnt.wu_sprung_x(E))
        ax.plot([-xe, xe], [E, E], color=plt.cm.plasma(0.15 + 0.8 * k / 60), lw=0.8)
    glow(ax, x, V, "#9ad0ff", lw=1.0)
    ax.set_xlim(-30, 30); ax.set_ylim(0, 165)
    save(fig, "wu-sprung")


# ------------------------------------------------------------------ Part V
@art("finite-differences", "Probability densities $|\\psi_n(x)|^2$ of the first 150 levels of the discretized Wu--Sprung Hamiltonian, one row per level.")
def _():
    fig, ax = canvas()
    L = qmnt.L_of(400); x = qmnt.grid(L, 1500)
    E, psi = qmnt.fd_levels(qmnt.wu_sprung_V(x), L, 150, return_vectors=True)
    ax.imshow((psi.T ** 2) ** 0.5, aspect="auto", cmap="magma", origin="lower")
    save(fig, "finite-differences")


@art("fitting", "Levenberg--Marquardt descending a curved valley: the path bends between careful gradient steps and bold Gauss--Newton jumps.")
def _():
    fig, ax = canvas()
    f = lambda x, y: (1 - x) ** 2 + 30 * (y - x ** 2) ** 2
    X, Y = np.meshgrid(np.linspace(-2.6, 2.6, 900), np.linspace(-0.9, 3.0, 340))
    ax.contourf(X, Y, np.log1p(f(X, Y)), 40, cmap="viridis", alpha=0.85)
    p = np.array([-2.2, 2.6]); lam = 1.0; path = [p.copy()]
    for _ in range(80):
        x, y = p; r = np.array([1 - x, np.sqrt(30) * (y - x ** 2)])
        J = np.array([[-1, 0], [-2 * np.sqrt(30) * x, np.sqrt(30)]])
        step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J)), -J.T @ r)
        if f(*(p + step)) < f(*p):
            p = p + step; lam *= 0.5
        else:
            lam *= 4
        path.append(p.copy())
    path = np.array(path)
    glow(ax, path[:, 0], path[:, 1], "#ffffff", lw=0.9)
    ax.scatter(path[:, 0], path[:, 1], s=6, color="#ffb347", zorder=5)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.9, 3.0)
    save(fig, "fitting")


@art("generalization", "Memorization: a degree-15 polynomial passes through every training point (shaded window) and goes wild the moment it leaves them.")
def _():
    fig, ax = canvas()
    xs = np.linspace(0, 1, 16); ys = np.sin(2 * np.pi * 2 * xs) + 0.15 * rng.normal(size=16)
    xx = np.linspace(-0.12, 1.12, 1500)
    c = np.polynomial.polynomial.polyfit(xs, ys, 15)
    yy = np.polynomial.polynomial.polyval(xx, c)
    ax.axvspan(0, 1, color="#1b2a4a")
    glow(ax, xx, np.sin(4 * np.pi * xx), "#7fd4ff", lw=0.8, alpha=0.7)
    glow(ax, xx, np.clip(yy, -4, 4), "#ff6fa8", lw=1.0)
    ax.scatter(xs, ys, s=16, color="white", zorder=5)
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-3, 3)
    save(fig, "generalization")


@art("controls", "Forty phase-randomized surrogates (faint) of the real zero fluctuations (bright): same power spectrum, no arithmetic.")
def _():
    fig, ax = canvas()
    n = np.arange(1, 401); f = qmnt.fluct(qmnt.zeros(400), n)
    F = np.fft.rfft(f - f.mean())
    for i in range(40):
        ph = np.exp(2j * np.pi * rng.uniform(size=F.size)); ph[0] = 1
        ax.plot(n, np.fft.irfft(F * ph, n=400), color=plt.cm.cool(i / 40), lw=0.5, alpha=0.25)
    glow(ax, n, f - f.mean(), "#ffb347", lw=0.9)
    ax.set_xlim(1, 400); ax.set_ylim(-1.3, 1.3)
    save(fig, "controls")


@art("sparse-frequency", "The spectrum of the first 10\\,000 zeros, $|\\sum_n e^{i\\omega t_n}|$: sharp peaks stand at $\\omega=\\log 2,\\log 3,\\log 4,\\log 5,\\dots$")
def _():
    fig, ax = canvas()
    t = qmnt.zeros(10000); w = np.linspace(0.4, 3.6, 4000)
    A = np.abs(np.exp(1j * np.outer(w, t)).sum(1))
    A = A / np.median(A)
    ax.fill_between(w, 0, A, color="#ff6fa8", alpha=0.25)
    glow(ax, w, A, "#ffb347", lw=0.7)
    for q in (2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31):
        ax.axvline(np.log(q), color="#7fd4ff", lw=0.4, alpha=0.5)
    ax.set_xlim(0.4, 3.6); ax.set_ylim(0, np.percentile(A, 99.9) * 1.05)
    save(fig, "sparse-frequency")


# ------------------------------------------------------------------ Part VI
@art("paper-setup", "Zeros 370 to 460: true fluctuations (dots) and the six-prime explicit formula (line), which knows nothing of the zeros it predicts.")
def _():
    fig, ax = canvas()
    n = np.arange(370, 461); t = qmnt.zeros(460)[369:460]
    f = qmnt.fluct(t, n); o = qmnt.fluct(qmnt.oracle_zeros(n, qmnt.first_primes(6)), n)
    glow(ax, n, o, "#ffb347", lw=1.0)
    ax.scatter(n, f, s=10, color="#7fd4ff", zorder=5)
    ax.axvspan(400.5, 460, color="#1b2a4a", zorder=0)
    ax.set_xlim(370, 460); ax.set_ylim(-1.2, 1.2)
    save(fig, "paper-setup")


@art("paper-memorization", "The continuation probe in miniature: a flexible fit paints one window of $\\sin(t\\log 2)$ perfectly, then cannot continue it.")
def _():
    fig, ax = canvas()
    t = np.linspace(14, 260, 3000); y = np.sin(t * np.log(2))
    win = (t > 40) & (t < 170)
    c = np.polynomial.chebyshev.chebfit(t[win], y[win], 60)
    yy = np.polynomial.chebyshev.chebval(t, c)
    ax.axvspan(40, 170, color="#1b2a4a")
    glow(ax, t, y, "#7fd4ff", lw=0.8, alpha=0.7)
    glow(ax, t[(t > 34) & (t < 176)], np.clip(yy, -2.5, 2.5)[(t > 34) & (t < 176)], "#ff6fa8", lw=1.0)
    ax.set_xlim(14, 260); ax.set_ylim(-2.4, 2.4)
    save(fig, "paper-memorization")


@art("paper-arithmetic", "Predicted fluctuations of zeros 401 to 500 from 1, 3, 6 and 25 primes (bottom to top), converging on the truth (white).")
def _():
    fig, ax = canvas()
    n = np.arange(401, 501); f = qmnt.fluct(qmnt.zeros(500)[400:500], n)
    for i, k in enumerate((1, 3, 6, 25)):
        o = qmnt.fluct(qmnt.oracle_zeros(n, qmnt.first_primes(k)), n)
        off = 2.4 * i
        ax.plot(n, f + off, color="white", lw=0.6, alpha=0.6)
        glow(ax, n, o + off, plt.cm.spring(i / 4), lw=0.9)
    ax.set_xlim(401, 500); ax.set_ylim(-1.3, 8.5)
    save(fig, "paper-arithmetic")


@art("paper-spurious-floor", "Complex eigenvalues of $H+iA$ for a random real $A$ of growing strength: the spectrum lifts off the real axis and scatters.")
def _():
    fig, ax = canvas()
    n = 70; E0 = np.arange(1, n + 1) ** 1.3
    G = rng.normal(size=(n, n)); A = (G + G.T) / 2
    for k, a in enumerate(np.linspace(0, 3.5, 60)):
        ev = np.linalg.eigvals(np.diag(E0) + 1j * a * A)
        ax.scatter(ev.real, ev.imag, s=1.5, color=plt.cm.plasma(k / 60), alpha=0.7)
    ax.set_xlim(-2, E0[-1] * 0.85); ax.set_ylim(-28, 28)
    save(fig, "paper-spurious-floor")


@art("paper-critique", "Pair correlation of the first 10\\,000 zeros (bars) against Montgomery's $1-(\\sin\\pi u/\\pi u)^2$: a law no fitted potential was asked to learn.")
def _():
    fig, ax = canvas()
    t = qmnt.zeros(10000); u = qmnt.Nsm(t)
    d = (u[None, :] - u[:, None])[np.triu_indices(10000, 1)] if False else None
    diffs = []
    for k in range(1, 8):
        diffs.append(u[k:] - u[:-k])
    diffs = np.concatenate(diffs); diffs = diffs[diffs < 3]
    h, e = np.histogram(diffs, bins=90, range=(0, 3)); c = (e[1:] + e[:-1]) / 2
    h = h / (len(u) * (e[1] - e[0]))
    ax.bar(c, h, width=e[1] - e[0], color="#3a5a9a", alpha=0.85)
    uu = np.linspace(1e-3, 3, 600)
    glow(ax, uu, 1 - (np.sin(np.pi * uu) / (np.pi * uu)) ** 2, "#ffb347", lw=1.1)
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.35)
    save(fig, "paper-critique")


# ------------------------------------------------------------------ appendices
@art("ap-python", "Four thousand random points in the unit square; the fraction inside the quarter circle estimates $\\pi/4$. Five lines of Python.")
def _():
    fig, ax = canvas()
    p = rng.uniform(size=(4000, 2)); inside = (p ** 2).sum(1) < 1
    ax.scatter(p[inside, 0] * 2.67, p[inside, 1], s=1.5, color="#ffb347")
    ax.scatter(p[~inside, 0] * 2.67, p[~inside, 1], s=1.5, color="#3a5a9a")
    th = np.linspace(0, np.pi / 2, 300)
    ax.set_xlim(0, 2.67); ax.set_ylim(0, 1)
    save(fig, "ap-python")


@art("ap-notation", "The symbols of this book.")
def _():
    fig, ax = canvas()
    syms = [r"\zeta", r"\rho", r"\psi", r"\Lambda", r"\pi", r"\sigma", r"\omega", r"\Gamma", r"\Delta", r"\lambda", r"\mu",
            r"\theta", r"\sum", r"\int", r"\partial", r"\hbar", r"\infty", r"\log", r"\phi", r"\varepsilon"]
    for i in range(70):
        s = syms[i % len(syms)]
        ax.text(rng.uniform(0.02, 0.98), rng.uniform(0.05, 0.95), f"${s}$", fontsize=rng.uniform(8, 30),
                color=plt.cm.cool(rng.uniform()), alpha=rng.uniform(0.3, 0.95), ha="center", va="center")
    save(fig, "ap-notation")


@art("ap-claims", "How the 32 chapters lead into one another: an edge joins each chapter to the chapters it assumes.")
def _():
    fig, ax = canvas()
    deps = {2: [1], 3: [2], 4: [2, 3], 5: [2, 3, 4], 6: [5], 7: [2], 8: [7], 9: [7, 8], 10: [2, 5, 8], 11: [10], 12: [4, 10, 11],
            13: [8, 10, 11], 14: [13], 15: [3, 4], 16: [2, 3, 15], 17: [4, 16], 18: [5, 6, 15, 16, 17], 19: [3, 4],
            20: [8, 17, 19], 21: [10, 16, 17, 18, 20], 22: [11, 12, 17, 21], 23: [8, 11], 24: [9, 13, 23], 25: [9, 19, 24],
            26: [6, 19, 25], 27: [6, 9, 18, 25, 26], 28: list(range(1, 28)), 29: [28], 30: [28], 31: [28], 32: [28, 29, 30, 31]}
    xpos = lambda k: (k - 1) / 31
    for k, ds in deps.items():
        for d in ds:
            a, b = xpos(d), xpos(k); m = (a + b) / 2; r = (b - a) / 2
            th = np.linspace(0, np.pi, 60)
            ax.plot(m + r * np.cos(th), 0.12 + 1.6 * r * np.sin(th), color=plt.cm.cool(k / 32), lw=0.5, alpha=0.5)
    ax.scatter([xpos(k) for k in range(1, 33)], [0.12] * 32, s=14, color="white", zorder=5)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1)
    save(fig, "ap-claims")


@art("ap-numbers", "The first nontrivial zero, $\\tfrac12+14.134725141734693790\\ldots\\,i$.")
def _():
    fig, ax = canvas()
    digits = "14.134725141734693790457251983562470270784257115699243175685567460149963429809256764949010393171561"
    for i, ch in enumerate(digits[:60]):
        ax.text(0.02 + (i % 30) * 0.0325, 0.68 - (i // 30) * 0.36, ch, fontsize=24, family="monospace",
                color=plt.cm.plasma(0.2 + 0.75 * i / 60), alpha=0.95 - 0.01 * i, ha="center", va="center")
    save(fig, "ap-numbers")


# ------------------------------------------------------------------ cover
def cover():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG); ax.set_axis_off()
    ax.set_xlim(-36, 36); ax.set_ylim(0, 1.2)
    # background texture: Sacks spiral of primes, very faint
    N = 30000; n = np.arange(1, N + 1); r = np.sqrt(n); th = 2 * np.pi * r
    isp = np.zeros(N + 1, bool); isp[qmnt.primes_upto(N)] = True; isp = isp[1:]
    ax.scatter(r[isp] * np.cos(th[isp]) * 0.24, 0.46 + r[isp] * np.sin(th[isp]) * 0.0042, s=1.6,
               c=r[isp], cmap="cool", alpha=0.42, lw=0)
    # Wu--Sprung well, zeros as energy levels
    x = np.linspace(-60, 60, 4000); V = qmnt.wu_sprung_V(x)
    t = qmnt.zeros(150)
    Vs = lambda v: 0.16 + 0.55 * (v - qmnt.V0) / (t[-1] - qmnt.V0)
    top = Vs(t[-1]) + 0.004
    xin = x[V <= t[-1]]
    ax.fill_between(xin, Vs(qmnt.wu_sprung_V(xin)), top, color="#101a36", alpha=0.95)
    for k, tk in enumerate(t):
        xe = float(qmnt.wu_sprung_x(tk))
        col = plt.cm.plasma(0.12 + 0.83 * k / len(t))
        for lw, a in ((5, 0.05), (2, 0.14), (0.6, 0.95)):
            ax.plot([-xe, xe], [Vs(tk)] * 2, color=col, lw=lw, alpha=a)
    m = V <= t[-1] * 1.02
    for lw, a in ((12, 0.03), (6, 0.07), (3, 0.18), (1.3, 1.0)):
        ax.plot(x[m], Vs(V[m]), color="#9ad0ff", lw=lw, alpha=a)
    # prime waves streaming out of the top of the well
    tt = np.linspace(-36, 36, 5000)
    for i, p in enumerate((2, 3, 5, 7, 11, 13, 17)):
        y = top + 0.035 + 0.016 * i + 0.010 * np.sin(tt * np.log(p) * 0.85 + i) * 3 / np.sqrt(p)
        for lw, a in ((4, 0.06), (0.9, 0.7)):
            ax.plot(tt, y, color=plt.cm.cool(i / 7), lw=lw, alpha=a)
    fig.savefig("cover.png", dpi=200, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or list(ART) + ["cover"]
    for s in which:
        if s == "cover":
            cover()
        else:
            ART[s]()
        print("done", s, flush=True)
    json.dump(CAPTION, open("captions.json", "w"), indent=1)
