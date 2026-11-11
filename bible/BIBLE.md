# Book bible

Working title: **Potentials, Primes and Prediction: A First Course on the Riemann Zeros, Quantum Spectra and Learning from Data**

Companion to: D. Soni, "Fitted potentials on the Wu--Sprung background memorize the Riemann zeros while the explicit formula predicts them", revision R1 (November 2026), AIP Advances manuscript ADV26-AR-04554, code and data at https://doi.org/10.5281/zenodo.24035819. Cited in the book as `\cite{soniR1}` and called "the paper".

This document is the contract for every chapter. Each chapter is written independently and in parallel, so everything a chapter may assume, define, call, cite or label is fixed here. Do not invent anything this document forbids.

---

## 1. Reader and goal

**Reader.** Has completed a first-year university physics course at the level of Halliday, Resnick and Walker (HRW), *Fundamentals of Physics*. Concretely the reader knows:

- single-variable calculus: limits, derivatives, chain/product rules, integrals, substitution, integration by parts, exponentials, logarithms, trigonometric functions and identities;
- vectors in 2D and 3D: components, dot product, magnitude;
- Newtonian mechanics, energy conservation, potential energy curves and turning points, simple harmonic motion, waves on a string, standing waves and harmonics;
- elementary electromagnetism (not needed here).

The reader does **not** know: complex numbers beyond hearing of $i$, infinite series beyond the geometric series, matrices beyond 2x2 arithmetic, eigenvalues, differential equations beyond SHM, quantum mechanics, probability beyond the average, number theory beyond the definition of a prime, or programming. Python is taught in Appendix A; code in chapters must be readable by someone who has read Appendix A.

**Goal.** After the book, the reader can read the paper line by line and understand every equation, table, figure, method, and claim, including its limitations, and can rerun its key computations at small scale.

**Dual use.** The book is a self-contained textbook on its topics *and* uses the paper as a running case study. Every chapter teaches its subject in general first and then connects it to the paper in a dedicated section. A reader who never opens the paper should still learn complex numbers, Fourier analysis, linear algebra, quantum mechanics, the zeta function, statistics and fitting.

---

## 2. Style: the two models

**Halliday, Resnick and Walker** for voice and pedagogy: friendly, concrete, physical, with motivating openers, Key Ideas, worked Sample Problems, Checkpoints, and graded end-of-chapter problems.

**Riley, Hobson and Bence** for completeness of argument: **no logical jumps**. Concretely:

1. Never write "it can be shown", "clearly", "obviously", "it is easy to see", "after some algebra", "one can verify", "standard manipulation". Show the step, or say exactly which earlier result (chapter and equation) it uses, or, for a result whose proof is genuinely beyond the book, say so explicitly in a labelled box (`\begin{beyond}`) stating what is being taken on trust and where the proof can be found.
2. Every symbol is defined in words the first time it appears in the chapter, even if defined in an earlier chapter (a short reminder: "recall from Chapter~\ref{ch:eigen} that ...").
3. Multi-step derivations show every intermediate line. If a line combines two operations, split it. Annotate non-obvious lines on the right with `\quad\text{(reason)}`.
4. Every new concept gets: a plain-language motivation, a precise definition, at least one fully worked numerical example, and one picture or table where a picture helps.
5. Distinguish status explicitly: **theorem (proved)**, **proved but proof beyond this book**, **heuristic/approximate**, **conjecture/open**, **numerical observation**. The Riemann hypothesis and the Hilbert--P\'olya conjecture are open; never imply otherwise.
6. When an approximation is made, state the condition under which it holds and what is neglected.

**Voice.** Second person ("you") and first person plural ("we") as in HRW. Short paragraphs. Physical analogies welcome. No hype, no filler, no motivational asides. British or American spelling consistently American (as in the paper).

**Punctuation rule (strict).** Never use an em dash: no `---` in LaTeX and no Unicode em dash. Use a colon, semicolon, comma or parentheses instead. En dashes `--` are allowed only for numeric ranges (`401--500`) and paired names (`Wu--Sprung`, `Riemann--von Mangoldt`).

**Length.** No upper limit. Typical chapter: 8,000 to 15,000 words. Length must come from complete derivations, examples and problems, never from repetition.

---

## 3. Chapter anatomy (mandatory, in this order)

```
\chapter{<Title>}\label{ch:<slug>}
\begin{chapterintro} ... \end{chapterintro}      % 1-3 paragraphs: what and why, ending with "where this shows up in the paper"
\begin{prerequisites} ... \end{prerequisites}    % bullet list: Chapter~\ref{ch:...} topics assumed; "first-year physics" items assumed
\section{...} ... (teaching sections)
   inside them, as often as useful:
   \begin{keyideas} \item ... \end{keyideas}       % at the end of each main section
   \begin{sample}{<short title>} ... \end{sample}  % worked problem: state, then numbered steps "Step 1:", "Step 2:", answer boxed with \boxed{}
   \begin{checkpoint} ... \end{checkpoint}          % quick conceptual question; answer given right after in \checkanswer{...}
   \begin{pitfall} ... \end{pitfall}                % common mistake
   \begin{beyond} ... \end{beyond}                   % result taken on trust; say where the proof is
   \begin{pycode} ... \end{pycode}                   % short Python listing (<= 30 lines)
\section{Paper connection}\label{sec:<slug>-paper}  % MANDATORY: connect to specific parts of the paper (section, equation, table, figure numbers of R1, see Section 8)
\section{Summary}                                    % bullet list of results with equation references
\section{Problems}                                   % 15-40 problems, graded: \prob{1}{...} easy, \prob{2}{...} medium, \prob{3}{...} hard
                                                     % at least 4 problems must use the paper (its numbers, tables or methods)
                                                     % at least 3 must be computational ("Using qmnt or numpy, ...")
\section{Answers to selected problems}               % answers (not just numbers: short solutions) to all \prob{1} and every odd-numbered problem
```

These environments and macros are defined in the book preamble; use exactly these names:

- `chapterintro`, `prerequisites` (itemize inside it), `keyideas` (itemize inside), `sample` (one argument: title), `checkpoint`, `\checkanswer{...}`, `pitfall`, `beyond`, `pycode` (verbatim Python, a `listings` environment), `\prob{difficulty 1|2|3}{text}` used inside `\begin{problemlist} ... \end{problemlist}`.
- Math macros available: `\Nsm`, `\Nfl`, `\gap`, `\dd` (upright d for differentials: `\dd x`), `\ii` (imaginary unit $i$; plain `i` also fine), `\ee` (Euler's e; plain `e` also fine), `\Re`, `\Im` (standard), `\abs{x}`, `\norm{x}`, `\braket{a}{b}`, `\bra{a}`, `\ket{a}`, `\rms`, `\corr`, `\order` ($\mathcal{O}$).
- Packages loaded: amsmath, amssymb, amsthm, mathtools, bm, graphicx, booktabs, hyperref, cleveref (use `\cref`), siunitx, xcolor, listings, tcolorbox. Do not add `\usepackage`.

**Labels.** Chapter label is given in Section 7 (`ch:<slug>`). Inside a chapter use only labels prefixed with the chapter slug: `sec:<slug>-...`, `eq:<slug>-...`, `fig:<slug>-...`, `tab:<slug>-...`. You may reference other chapters **only** through their chapter labels `\ref{ch:<slug>}` from Section 7 (never another chapter's section/equation labels, which you cannot see). To point at a specific result in another chapter, name it in words: "the Euler product (Chapter~\ref{ch:zeta})".

**Citations.** Use `\cite{key}` only with keys from Section 9. Do not cite anything else. If you would need another source, describe it in words without a citation.

---

## 4. Figures and code (machine-checked)

Your response has exactly three parts, in this order, separated by the marker lines shown (each marker alone on its own line):

```
=====CHAPTER=====
<the complete LaTeX for the chapter, starting with \chapter>
=====FIGURES=====
<zero or more figure scripts>
=====CHECKS=====
<one Python script>
```

**Figures.** Include a figure in the chapter as

```
\begin{figure}[htbp]\centering
\includegraphics[width=0.85\linewidth]{figs/<slug>_<name>.pdf}
\caption{<self-contained caption>}\label{fig:<slug>-<name>}
\end{figure}
```

and for **each** such figure give a script in the FIGURES part:

```
### FIG <slug>_<name>
```python
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import qmnt   # optional
... 
plt.savefig("<slug>_<name>.pdf", bbox_inches="tight")
```
```

Rules: the script runs from the `figs/` directory, writes exactly that filename, uses only numpy, scipy, matplotlib, mpmath and `qmnt` (Section 5), runs in under 3 minutes, uses no seaborn and no custom fonts, and uses `figsize=(6,4)` or `(6.5,3)` (two panels). Label axes with units. Aim for 3 to 8 figures per chapter; hand-drawn-style schematics are fine as matplotlib drawings.

**Checks.** The CHECKS part is a single Python script that recomputes **every numerical answer** in the chapter's Sample Problems and in the Answers section, plus every number you quote from your own computation, with `assert abs(x - claimed) < tol` (or `np.isclose`) and a comment naming the sample problem or problem number. It may use numpy, scipy, mpmath and `qmnt`. Numbers quoted from the paper are not rechecked here unless qmnt can reproduce them cheaply (it can for: zeros, smooth zeros, gaps, nulls, oracle rows of Table V for holdouts up to 2000, Wu--Sprung $L$ values). The script must run in under 5 minutes. If a number cannot be checked by code (a pure algebra result), check a numerical instance of it.

Do not invent numerical values. Every number in the chapter is either (a) computed in CHECKS, (b) quoted from the paper with its location, or (c) a standard constant or textbook value (state it).

---

## 5. The `qmnt` companion library (exists; you may call it)

`import qmnt` provides (all arrays are numpy; indices $n$ start at 1 as in the paper):

| Function | Meaning |
|---|---|
| `qmnt.zeros(nmax)` | imaginary parts $t_1,\dots,t_{n_{\max}}$ of the first nontrivial zeta zeros, $n_{\max}\le 10000$ (cached; $t_1=14.134725\ldots$) |
| `qmnt.Nsm(t)` | smooth counting function, paper Eq. (1) |
| `qmnt.smooth_zeros(n)` | $s_n$ solving $N_{\rm sm}(s_n)=n-\tfrac12$ |
| `qmnt.mean_gap(n)` | $\mathrm{gap}_n = ds_n/dn = 2\pi/\log(s_n/2\pi)$ |
| `qmnt.fluct(t, n)` | $(t-s_n)/\mathrm{gap}_n$, fluctuation in mean spacings |
| `qmnt.primes_upto(x)`, `qmnt.first_primes(k)` | primes |
| `qmnt.Nfl(t, primes, mmax=3)` | truncated explicit-formula fluctuation, paper Eq. (4) |
| `qmnt.oracle_zeros(n, primes, mmax=3)` | zeros predicted by solving $N_{\rm sm}(t)+N_{\rm fl}(t)=n-\tfrac12$ (Brent, bracket $\pm3$ gaps around $s_n$) |
| `qmnt.rms(a)`, `qmnt.corr(a, b)` | root mean square; Pearson correlation |
| `qmnt.score(t_pred, n)` | returns `(rms, corr)` of predicted vs true fluctuations for indices `n`, in mean spacings (the paper's scoring) |
| `qmnt.V0` | $3.10073\pi$ |
| `qmnt.wu_sprung_x(V)` | paper Eq. (2) |
| `qmnt.wu_sprung_V(x)` | inverse of Eq. (2) (Brent), even in $x$; returns $V_0$ at $x=0$ |
| `qmnt.L_of(Vmax)` | $x(V_{\max})$: 44.883 for 1000, 59.957 for 1500 |
| `qmnt.fd_levels(Vvals, L, k=None, return_vectors=False)` | eigenvalues (and vectors) of the finite-difference Hamiltonian $-d^2/dx^2+V$ on the $N=$`len(Vvals)` interior points of $(-L,L)$ with Dirichlet walls; first `k` levels |
| `qmnt.grid(L, N)` | the $N$ interior points |
| `qmnt.richardson(E_h, E_h2)` | $(4E_{h/2}-E_h)/3$ |
| `qmnt.fit_potential(nfit, M, basis="cosine", ridge=0.0, scale=1, holdout=None, maxiter=60)` | the paper's Hermitian fit (Sec. II C, App. A step 6) at reduced cost; returns dict with keys `c`, `insample_rms`, `oos_rms`, `oos_corr`, `E` (fitted levels). Slow (minutes); use in figures/checks only if essential, never more than one call per script |

Units everywhere: $\hbar = 2m = 1$, so $H=-d^2/dx^2+V(x)$ and energies equal the "heights" $t$.

---

## 6. Notation (matches the paper; use exactly)

| Symbol | Meaning |
|---|---|
| $\rho_n = \tfrac12 + i t_n$ | $n$-th nontrivial zeta zero (assuming it is on the critical line, which is verified numerically for all zeros used); $t_n>0$ its height |
| $N(T)$ | number of zeros with $0<t\le T$ |
| $N_{\rm sm}(t)$ (`\Nsm`) | smooth part, paper Eq. (1): $\frac{t}{2\pi}\log\frac{t}{2\pi}-\frac{t}{2\pi}+\frac78$ |
| $S(t)$ | fluctuating part $N(t)-N_{\rm sm}(t)$ (up to the $\mathcal O(1/t)$ term) |
| $N_{\rm fl}(t)$ (`\Nfl`) | truncated prime sum, paper Eq. (4) |
| $s_n$ | smooth zeros, $N_{\rm sm}(s_n)=n-\tfrac12$ |
| $\mathrm{gap}_n$ (`\gap_n`) | local mean spacing $ds_n/dn$ |
| $p$, $p^m$ | prime, prime power; $\log$ is always natural log |
| $\Lambda(k)$ | von Mangoldt function |
| $H=-\frac{d^2}{dx^2}+V(x)$ | Hamiltonian, units $\hbar=2m=1$ |
| $V_0 = 3.10073\pi$ | bottom of the Wu--Sprung potential |
| $L$ | half-width of the box $[-L,L]$ |
| $N$ (grid) | number of interior grid points; $h=2L/(N+1)$ spacing. (Context distinguishes from the counting function.) |
| $E_n$, $\psi_n$ | eigenvalues, normalized eigenfunctions |
| $\delta V(x)=\sum_{k=0}^{M-1}c_k\phi_k(x)$ | fitted correction; $M$ basis size, $c_k$ coefficients, $\phi_k$ basis functions ($\cos(k\pi x/L)$ by default) |
| $A_{nk}=\int|\psi_n|^2\phi_k\,dx$ | first-order sensitivity matrix |
| $r_n=\tilde t_n - E_n$ | residuals |
| $\tilde t_n$, $\Delta_n$, $E^{\rm Rich}$ | bias-corrected targets, grid bias, Richardson extrapolant |
| $\lambda$ | Levenberg--Marquardt damping; $\mu$ ridge strength |
| $iA$, $A_{nm}=\langle n|A|m\rangle$, $T^{(n)}$, $\Gamma$, $R$ | non-Hermitian perturbation, its matrix elements, second-order tensors, widths, amplitude-ball radius |
| $\omega$, $T$ (window), $K$ | angular frequency in $t$, length of training window in $t$, number of selected frequencies |
| "rms" | root mean square error in units of local mean spacing; "corr" Pearson correlation of predicted vs true fluctuations |
| null | the prediction "no fluctuation" ($t_n \approx s_n$, or unperturbed potential levels) |
| OOS | out-of-sample (held-out) |

**Overloaded symbols in the paper** (point these out where relevant): $A$ is used for the sensitivity matrix $A_{nk}$ and for the non-Hermitian operator $A$; $T$ for the training window, the tensors $T^{(n)}$, and Chebyshev polynomials $T_k$; $N$ for the grid size and the counting function.

---

## 7. Chapter map (contracts)

Each entry: **number, slug, title**. *Assumes*: what the chapter may use without explanation (always: first-year physics). *Must teach*: the minimum content. *Paper hooks*: what in the paper this chapter enables. Chapters may teach more, but must not assume more.

### Part I. Mathematical foundations

**1. `intro` The Question: Can a Fitted Potential Learn the Riemann Zeros?**
Assumes: nothing beyond first-year physics. Must teach (informally, almost no equations): a list of mysterious numbers $t_1=14.13, t_2=21.02,\dots$; that they come from a function built from primes (details later); that they look like energy levels of a quantum system (Hilbert--P\'olya idea, as a hope not a fact); what "fitting a model" means (a ruler-and-points analogy); memorizing vs predicting (a student who memorizes past exam answers); training vs held-out data; the paper's result in one page in words; a road map of the book with a dependency diagram of the six parts; how to use the book (Sample Problems, Checkpoints, Problems, code; Appendix A for Python). Paper hooks: abstract, Fig. 1, Sec. I.

**2. `complex` Complex Numbers and the Exponential**
Assumes: Ch. 1. Must teach: $i$, arithmetic, complex plane, modulus, argument, conjugate, polar form; Euler's formula derived from Taylor series (state Taylor series of $e^x,\sin,\cos$ as known from calculus, or derive briefly); $e^{i\theta}$ as rotation; $\cos\theta=(e^{i\theta}+e^{-i\theta})/2$; powers $a^{s}=e^{s\log a}$ for complex $s$, so $p^{-s}=p^{-\sigma}e^{-it\log p}$ with $s=\sigma+it$ (crucial); $|p^{-s}|=p^{-\sigma}$; phase rotating at angular frequency $\log p$ in $t$; complex functions of a complex variable; zeros of a complex function (both real and imaginary parts vanish); argument (phase) of a complex number and of a product; logarithm of a complex number (branches, informally). Paper hooks: $\rho_n=\tfrac12+it_n$, $\sin(mt\log p)$ in Eq. (4).

**3. `series` Sequences, Series and Infinite Products**
Assumes: Ch. 2. Must teach: sequences, limits; partial sums; geometric series; harmonic series diverges (with proof); $\sum 1/k^s$ for real $s$: converges iff $s>1$ (integral test proved); absolute convergence; complex series and $\sum k^{-s}$ converges absolutely for $\Re s>1$; conditional convergence (alternating series); Taylor series of $\log(1-x)$ and radius of convergence; infinite products, $\log$ of a product is a sum of logs; when an infinite product converges; telescoping. Paper hooks: preparation for Euler product and explicit formula; why truncating a sum over primes is an approximation.

**4. `asymptotics` Approximation, Asymptotics and Root Finding**
Assumes: Ch. 2--3. Must teach: big-O, little-o, $\sim$; leading-order behavior; comparing sums to integrals; Stirling's formula for $\log n!$ and $\log\Gamma$ (state $\Gamma$ as generalized factorial; Stirling for complex argument stated in a `beyond` box); inverting a monotone function; root finding: bisection (with error bound), Newton, the idea of Brent's method (bracketing + fast steps), bracketing failures and bracket expansion; error, relative error, significant digits; floating-point numbers, machine epsilon ($2.2\times10^{-16}$ double, $1.2\times10^{-7}$ float32) and why "analytically zero" can become $10^{-7}$ noise. Paper hooks: solving $N_{\rm sm}(s_n)=n-\tfrac12$; Brent in App. A; float32 noise artifact in Sec. V; Lambert-$W$ mention in Sec. I A (define $W$ briefly).

**5. `fourier-series` Oscillations, Superposition and Fourier Series**
Assumes: Ch. 2--4, SHM and standing waves. Must teach: sinusoids, amplitude, phase, angular frequency $\omega$, period $2\pi/\omega$; superposition; beats; orthogonality of sines/cosines over a period (derived); Fourier series coefficients (derived from orthogonality); complex form; Parseval; even/odd functions and cosine/sine series; the cosine basis $\cos(k\pi x/L)$ on $[-L,L]$ for even functions; Gibbs phenomenon briefly; quasi-periodic (almost periodic) sums of incommensurate frequencies like $\sin(t\log2)+\sin(t\log3)$, which never repeat; why such a function cannot be represented by any finite window's periodic extension. Paper hooks: cosine basis of $\delta V$; Eq. (4) as almost-periodic sum; the $p=2$ period $2\pi/\log2\approx9.06$ in $t$; the continuation probe.

**6. `fourier-transform` The Fourier Transform, Sampling and the Periodogram**
Assumes: Ch. 5. Must teach: from series to transform; transform of a finite-length sinusoid (sinc), resolution $\Delta\omega\approx2\pi/T$ derived; windows and leakage; sampling at irregular points (the zeros are irregular samples in $t$); discrete Fourier transform and FFT (what it computes); periodogram and peak finding; power spectrum; autocorrelation and the Wiener--Khinchin connection (stated, with discrete example); phase randomization: keeps $|$coefficients$|$, scrambles phases; what that preserves (power spectrum, autocorrelation) and destroys. Paper hooks: Sec. II D controls C1, C2 ("half the Fourier resolution"); surrogate tests (Sec. III); Landau--Gonek peaks at $\log p^m$ (Sec. IV); App. A step 20, 23.

**7. `linear-algebra` Vectors, Matrices and Linear Systems**
Assumes: Ch. 2, 3D vectors. Must teach: vectors in $n$ dimensions (real and complex), inner product, norm, orthogonality, orthonormal basis, projection; matrices as linear maps; multiplication, transpose, conjugate transpose; identity, inverse; solving $Ax=b$ by elimination; rank; overdetermined vs underdetermined systems; determinant (2x2, 3x3, properties stated); functions as vectors: sampling a function on a grid; $\int f g\,dx$ as inner product; basis expansion of a function. Paper hooks: $A_{nk}$ as a matrix; $c$ as a vector; residual vector; "more coefficients than levels".

**8. `eigen` Eigenvalues, Eigenvectors and Symmetric Matrices**
Assumes: Ch. 7. Must teach: eigen-equation, characteristic polynomial for 2x2 and 3x3; symmetric real and Hermitian matrices have real eigenvalues and orthogonal eigenvectors (proved); spectral theorem (proved for distinct eigenvalues; general case in `beyond`); diagonalization; tridiagonal matrices; non-symmetric matrices can have complex eigenvalues (example); the second-derivative operator as a matrix; eigenvalue problems for differential operators (string with fixed ends: $-y''=\lambda y$, eigenvalues $(k\pi/2L)^2$); numerical eigensolvers as black boxes (what LAPACK does, cost $\mathcal O(N^3)$, tridiagonal $\mathcal O(N^2)$), `scipy.linalg.eigh_tridiagonal`; eigenvalue ordering and matching eigenvectors by overlap. Paper hooks: Sec. II B ("eigenpairs from LAPACK"), Sec. II E (iii) matching by overlap, "self-adjoint" in Hilbert--P\'olya.

**9. `least-squares` Least Squares, the SVD and Regularization**
Assumes: Ch. 7--8. Must teach: fitting a line (derive normal equations); linear least squares in general; geometric picture (projection); normal equations; singular value decomposition (existence in `beyond`, meaning: rotations and stretches; relation to eigen of $A^\top A$); pseudo-inverse; condition number; noise amplification by small singular values; truncated SVD; ridge (Tikhonov) regularization derived, filter factors $\sigma/(\sigma^2+\lambda\sigma_1^2)$ (paper's form, with $\sigma_1$ the largest singular value); underdetermined problems have infinitely many exact fits; minimum-norm solution; a polynomial-fitting example showing overfitting (forward reference to Ch. 25). Paper hooks: App. A step 6 SVD-damped step, ridge variant $\mu$; Sec. IV control B "underdetermined" at 900 coefficients on 400 zeros.

### Part II. Quantum mechanics of a particle on a line

**10. `schrodinger` From Waves to Wavefunctions**
Assumes: Ch. 2, 5, 8, waves on a string. Must teach: de Broglie relation; why a wave equation; time-dependent Schr\"odinger equation (motivated, not derived from deeper principles; say so); separation of variables; time-independent equation $H\psi=E\psi$; Hamiltonian as an operator; Born rule, normalization; operators and expectation values; Hermitian (self-adjoint) operators have real eigenvalues (link to Ch. 8); units $\hbar=2m=1$ and why the paper uses them; energy levels as a spectrum. Paper hooks: Sec. II B "$H=-d^2/dx^2+V$", the phrase "spectrum of a self-adjoint operator" in Sec. I.

**11. `bound-states` Bound States in One Dimension**
Assumes: Ch. 10. Must teach: infinite square well (box $[-L,L]$, Dirichlet walls): solve fully, levels $E_k=(k\pi/2L)^2$ in these units; normalization; node counting; parity for even potentials (even/odd states alternate); harmonic oscillator levels (solve by series or state with check); qualitative shape of states in a general confining potential; energies grow, spacing shrinks or grows depending on potential shape (linear well, $V\propto|x|$, logarithmic shapes); level density; why a potential that grows slowly gives levels that crowd together. Paper hooks: Dirichlet walls at $\pm L$; "first 500 levels scored"; why the Wu--Sprung potential is even.

**12. `wkb` The Semiclassical Approximation: WKB and Counting Levels**
Assumes: Ch. 4, 10--11, classical turning points. Must teach: WKB ansatz, local wavenumber $k(x)=\sqrt{E-V(x)}$ (units $\hbar=2m=1$); turning points; connection formulas (state, with the $\pi/4$ phase, proof in `beyond`); Bohr--Sommerfeld quantization $\int_{x_1}^{x_2}\sqrt{E-V}\,dx=(n-\tfrac12)\pi$; phase-space area; counting function $N(E)=\frac1\pi\int\sqrt{E-V}\,dx+\tfrac12$ (derive); accuracy for high levels, weakness at low levels; turning point $x_E$ maps each level to a position (level $n$ probes the potential most strongly near its turning points); inversion: given $N(E)$, recover $x(V)$ for an even, monotone potential via an Abel integral equation (derive the Abel inversion formula with full steps). Paper hooks: Eq. (2) derivation is in Ch. 22 using this; "position maps to level index through the classical turning point" (Sec. III, interpolation split); "known semiclassical weakness at low $n$".

**13. `perturbation` Perturbation Theory**
Assumes: Ch. 8, 10--11. Must teach: first-order energy shift $\delta E_n=\langle n|\delta V|n\rangle=\int|\psi_n|^2\delta V\,dx$ (derive); linearization: $\delta E_n\approx\sum_k A_{nk}c_k$ with $A_{nk}=\int|\psi_n|^2\phi_k$; second-order shift $\sum_{m\ne n}|V_{nm}|^2/(E_n-E_m)$ (derive); level repulsion; worked examples in the infinite well (constant shift, linear tilt, cosine perturbation); validity condition (matrix elements small compared to level spacings); compare with exact numerical diagonalization; why "fitting by iterated linearization" works (Gauss--Newton preview). Paper hooks: Sec. II C sensitivity $A_{nk}$; Eq. (5); Sec. V "left the perturbative regime".

**14. `non-hermitian` Beyond Hermitian: Complex Potentials and PT Symmetry**
Assumes: Ch. 13. Must teach: complex potential $V-i\Gamma/2$ and decay (norm decreases), complex eigenvalues; $H+iA$ with $A$ real multiplication operator; first order: shift $i A_{nn}$ purely imaginary; second order real shift $-\sum_{m\ne n}|A_{nm}|^2/(E_n-E_m)$ (derive from Ch. 13 with $V\to iA$, careful with $i^2$); quadratic form $-c^\top T^{(n)}c$; antisymmetry of pair contributions and the sum rule $\sum_n\delta E_n=0$ over a complete truncated basis (prove); locality of the redistribution; PT symmetry: parity and time reversal, odd imaginary potentials, real spectra to leading order, symmetry breaking (state, example $2\times2$ matrix); constant $iA$ shifts all levels by an imaginary constant with zero real second-order shift (the "null mode"); breakdown of perturbation theory at large amplitude; exact diagonalization of complex non-symmetric matrices; matching eigenstates by overlap. Paper hooks: Sec. II E, Sec. V, Fig. 5, Sec. V A.

### Part III. Primes and the zeta function

**15. `primes` The Primes**
Assumes: Ch. 3--4. Must teach: definition, fundamental theorem of arithmetic (prove existence, uniqueness sketch with honest labeling), infinitely many primes (Euclid), sieve of Eratosthenes, prime counting $\pi(x)$, prime number theorem (statement, history, proof beyond), $\pi(x)\sim x/\log x$ and $\mathrm{li}(x)$; prime powers; von Mangoldt $\Lambda(k)$; Chebyshev $\psi(x)=\sum_{k\le x}\Lambda(k)\sim x$; numerical tables; $\log p$ as the natural "size" of a prime. Paper hooks: "primes $p\le13$", "prime powers up to 113 except 32, 64, 81", $\log p^m$.

**16. `zeta` The Riemann Zeta Function**
Assumes: Ch. 2--3, 15. Must teach: $\zeta(s)=\sum k^{-s}$ for $s>1$, then complex $\Re s>1$; Euler product (prove from unique factorization, with convergence); divergence at $s=1$ implies infinitely many primes; analytic continuation explained honestly (the idea of extending a function beyond where its formula converges, illustrated with the geometric series $1/(1-z)$; then the alternating zeta $\eta(s)=(1-2^{1-s})\zeta(s)$ extends $\zeta$ to $\Re s>0$, shown fully); the functional equation (stated in `beyond`); the pole at $s=1$; trivial zeros; critical strip and critical line; nontrivial zeros come in symmetric pairs; RH stated precisely, status (open; verified for more than $10^{13}$ zeros, state as reported fact); computing zeros: `mpmath.zetazero(n)`; table of first 10 zeros; the Riemann--Siegel $Z$ function as a real function whose sign changes are zeros (stated). Paper hooks: Sec. I first sentence; Sec. II A data; "$\rho_n=\tfrac12+it_n$".

**17. `counting-zeros` Counting the Zeros**
Assumes: Ch. 4, 16. Must teach: $N(T)$; argument principle (intuitive: counting how many times the phase winds; full theorem in `beyond`); Riemann--von Mangoldt: $N(T)=\frac{\theta(T)}{\pi}+1+S(T)$ with $\theta$ the Riemann--Siegel theta function; derive $\theta(T)\approx\frac T2\log\frac{T}{2\pi}-\frac T2-\frac\pi8$ from Stirling (Ch. 4) step by step; hence $N_{\rm sm}(t)$ of paper Eq. (1) with the $7/8$; $S(T)=\frac1\pi\arg\zeta(\tfrac12+iT)$, small and oscillating, $|S|$ typically below 1 at these heights; smooth zeros $s_n$; mean spacing $\mathrm{gap}_n=2\pi/\log(s_n/2\pi)$ (derive); unfolding; fluctuations in mean spacings; why errors are quoted in mean spacings; numerical illustration with qmnt. Paper hooks: Eq. (1), Sec. II A, every rms in the paper; the null 0.263.

**18. `explicit-formula` The Explicit Formula: Primes as Frequencies**
Assumes: Ch. 5--6, 15--17. Must teach: $\log\zeta(s)=\sum_p\sum_m\frac{p^{-ms}}{m}$ for $\Re s>1$ (derive from Euler product and $-\log(1-x)$ series); taking imaginary part on the line $s=\tfrac12+it$ formally gives $S(t)\approx-\frac1\pi\sum_p\sum_m\frac{\sin(mt\log p)}{mp^{m/2}}$ (derive the formal step, then explain honestly that the series does not converge on the critical line and why truncation still works as an approximation: smoothing, heuristic status, references to rigorous versions in `beyond`); paper Eq. (4); predicting zero $n$ by solving $N_{\rm sm}+N_{\rm fl}=n-\tfrac12$; how many primes; accuracy vs number of primes (reproduce small part of Table V with qmnt); the Riemann--von Mangoldt/Guinand--Weil explicit formula stated as a duality between zeros and prime powers; Landau's formula: $\sum_{0<t_n\le T}x^{\rho_n}\approx-\frac{T}{2\pi}\Lambda(x)$ (stated), so the "spectrum of the zeros" has peaks at $\log p^m$; Fran\c ca--LeClair; Connes' truncated Weil form (mention only). Paper hooks: Sec. II D, Eq. (4), Table V, Sec. IV, refs 16--18, 25--26.

### Part IV. Spectra, randomness and operators

**19. `probability` Probability and Statistics Essentials**
Assumes: Ch. 3--4. Must teach: random variables, distributions (discrete, continuous, densities), mean, variance, standard deviation, rms (and why rms of an error vs standard deviation), histograms, the Gaussian, exponential distribution (Poisson process gaps), covariance, Pearson correlation (derive range $[-1,1]$ via Cauchy--Schwarz), correlation of a prediction with truth vs its rms error (they measure different things; a prediction with correct shape but wrong amplitude), autocorrelation of a sequence; sampling and estimators; standard error; confidence intervals (intro); law of large numbers (stated); random number generators and seeds (reproducibility). Paper hooks: every "rms / corr" pair; "seeds"; "OOS rms worse than null but corr near zero".

**20. `level-statistics` Level Statistics and Random Matrices**
Assumes: Ch. 8, 17, 19. Must teach: nearest-neighbor spacings; unfolding to unit mean; Poisson statistics for uncorrelated levels (derive $e^{-s}$); level repulsion; $2\times2$ real symmetric and complex Hermitian random matrices: derive the spacing densities $\propto s$ and $\propto s^2$ and normalize to Wigner surmises, including $\frac{32}{\pi^2}s^2e^{-4s^2/\pi}$ (full derivation); GOE/GUE names; Montgomery's pair correlation $1-(\sin\pi u/\pi u)^2$ and Dyson's remark (history); Odlyzko's numerics; reproduce paper Fig. 2 with qmnt; what GUE statistics do and do not imply (no specific operator); short-range negative correlations of consecutive spacings (connect to Sec. III's negative autocorrelation at lags 1--3). Paper hooks: Fig. 2, Sec. I refs 1--2, within-height splits discussion.

**21. `hilbert-polya` The Hilbert--P\'olya Dream**
Assumes: Ch. 10, 16--18, 20. Must teach: the conjecture and why a self-adjoint operator would prove RH (real eigenvalues $\Rightarrow$ zeros on the line); the trace-formula analogy: Gutzwiller (periodic orbits) vs explicit formula (primes), orbit lengths $\leftrightarrow\log p$, repetitions $\leftrightarrow m$, stability factor $\leftrightarrow p^{-m/2}$ (as an analogy, carefully labeled); Berry--Keating $xp$ Hamiltonian and its semiclassical counting reproducing $N_{\rm sm}$ to leading order (show); Connes' approach (description); Bender--Brody--M\"uller and Bellissard's comment (description, no verdict beyond what is published); Sierra; status: no operator known; analytic program vs the empirical (fitting) program. Paper hooks: Sec. I A d "Operator proposals", refs 3--11, Sec. VI first paragraph.

**22. `wu-sprung` Building a Potential from the Zeros: the Wu--Sprung Construction**
Assumes: Ch. 11--12, 17, 21. Must teach: the inverse spectral problem in general (non-uniqueness; why one fixes evenness and monotonicity); setting the WKB counting function of an even potential equal to $N_{\rm sm}(E)$ and solving the Abel equation of Ch. 12 to derive paper Eq. (2) **step by step**, including where $V_0$ comes from (the condition that fixes the constant; explain the paper's numerical $V_0=3.10073\pi$ as the value that makes the construction consistent, and check numerically with qmnt that WKB levels of the resulting potential track $s_n$); the shape of the potential ($x\sim\sqrt V\log V$, so $V$ grows roughly like $(x/\log x)^2$); $L=x(V_{\max})$ values 44.883 and 59.957; Wu and Sprung's fractal fluctuation corrections; van Zyl--Hutchinson, Slater, Jaksch; the empirical program's implicit logic and why it needs a held-out test. Paper hooks: Sec. II B, Eq. (2), Sec. I A a.

### Part V. Computation, fitting and generalization

**23. `finite-differences` Solving the Schr\"odinger Equation on a Computer**
Assumes: Ch. 8, 11. Must teach: discretizing $-\psi''$ with the second difference; error $\mathcal O(h^2)$ by Taylor (derive); Dirichlet walls and the tridiagonal matrix (diag $2/h^2+V_j$, off $-1/h^2$); infinite well test: exact discrete eigenvalues $\frac{4}{h^2}\sin^2\frac{k\pi h}{4L}$ (derive) vs continuum, error grows with level index (dispersion); levels 401--500 can be off by several mean spacings; Richardson extrapolation (derive $(4E_{h/2}-E_h)/3$); correcting targets instead of the operator; the paper's leakage argument Eq. (3) (algebra in full); normalization of discrete eigenvectors $\sum\psi^2h=1$. Paper hooks: Sec. II B, Eq. (3), App. A steps 3--5, "raw-zero null 2.09".

**24. `fitting` Fitting a Potential to a Spectrum**
Assumes: Ch. 9, 13, 23. Must teach: nonlinear least squares; Gauss--Newton from linearization ($A_{nk}$); Levenberg--Marquardt damping (derive as interpolation between Gauss--Newton and gradient descent); the paper's SVD-damped step $c\mathrel{+}=V\,\mathrm{diag}(\sigma/(\sigma^2+\lambda\sigma_1^2))U^\top r$ and try-list schedule (App. A step 6) explained line by line; accept-only-improvements rule; random restarts; basis choices: cosine, Chebyshev $T_k(2|x|/L-1)$, hat functions (define all three, plot them); the coverage law: a cosine of wavenumber $k\pi/L$ can resolve structure on the scale of the local wavelength $\pi/\sqrt E$ only if $M\pi/L>2\sqrt E$ (derive heuristically); in-sample error falls with $M$. Paper hooks: Sec. II C, App. A step 6, Sec. III first paragraph, Table II.

**25. `generalization` Memorization and Generalization**
Assumes: Ch. 9, 19, 24. Must teach: what a model "learns"; training and test sets; in-sample vs out-of-sample error; the null model and why every claim is relative to it; overfitting with a worked polynomial example; capacity; interpolation vs extrapolation; covariate shift (test data from a different region); within-range random splits and odd/even splits; saturated models ($M\ge$ number of training points); serial correlation and why smooth interpolation can be anticorrelated with truth when consecutive values are negatively correlated (derive for a simple model); simple kriging / best linear predictor from autocovariance (derive normal equations, nugget); regularization and its effect (returns to null); a model that can fit anything has no inductive bias; inductive bias and the right parametrization. Paper hooks: Sec. II C OOS protocol, within-height splits, Tables I and III, Fig. 3, Fig. 4 discussion, Sec. VI.

**26. `controls` Uncertainty and Controls**
Assumes: Ch. 6, 19, 25. Must teach: sampling variability; bootstrap: idea, percentile intervals, worked example; failure under serial correlation; moving/circular block bootstrap; permutation tests and what null hypothesis they test; surrogate data: random permutation, phase-randomized (Theiler), IAAFT (Schreiber--Schmitz): what each preserves; using surrogates to test "is this fit learning structure or capacity?"; exploratory vs confirmatory analysis; statistics defined after seeing the data (post hoc) and why they cannot support significance claims; ensembles of replicates; reporting ranges over seeds vs standard errors. Paper hooks: bootstrap intervals in Secs. III--IV and Tables II, IV; surrogate tests Sec. III; App. A steps 18, 22, 23; the "score defined after the data were seen" qualification.

**27. `sparse-frequency` Learning Frequencies from Data**
Assumes: Ch. 6, 9, 18, 25--26. Must teach: sinusoidal regression with known frequencies (linear least squares; this is control B); dense uniform grid of frequencies (control C1) and why it behaves like a generic basis; greedy sparse frequency search: periodogram of residual, pick largest peak, refine, refit jointly, repeat (control C2), connection to matching pursuit; choosing $K$ and ridge by a forward validation split inside the training window; frequency estimation error $\sim 1/T$ and dephasing: an error $\delta\omega$ produces phase error $\delta\omega\,\Delta t$ growing with distance (derive), why skill decays far from the window; recovering $\log p^m$ from the zeros: the matching criterion $|\omega-\log q|<\pi/T$; why "sparse in frequency" is a strong prior; root ambiguity when the level equation $N_{\rm sm}+F=n-\tfrac12$ has several roots. Paper hooks: Sec. II D controls, Sec. IV, Table IV, App. A steps 20, 23.

### Part VI. The paper, section by section

These five chapters teach no new mathematics. They read the paper in order, quote its claims (paraphrase plus short quotes), and for each claim give: (a) what it says in plain words, (b) which chapters supply the tools, (c) the computation behind it, rerun at small scale with qmnt where feasible, (d) its strength and its limits. Each has problems that ask the reader to reproduce or vary a result. Be faithful: report the paper's numbers exactly; do not add claims the paper does not make; when the paper hedges, keep the hedge.

**28. `paper-setup` Reading the Paper I: The Question, the Data and the Protocol** (paper Sec. I, II A--D, II F, Figs. 1--2, App. A steps 1--8). Includes: the abstract unpacked sentence by sentence; the scope statement; relation to prior work (four strands) and the literature audit's basis and limits; the two nulls; the leakage argument; the AI-use disclosure and why reproducibility records matter.
Assumes: all of Parts I--V.

**29. `paper-memorization` Reading the Paper II: Fitting Is Easy, Transfer Is Nil** (Sec. III; Tables I--III; Figs. 1, 3, 4; App. A steps 12--18, 21--22). Includes: coverage, twelve configurations, bootstrap intervals, other bases and the marginal Chebyshev row, within-height splits, surrogates and kriging, interpolation-gap damage, the arithmetic-continuation probe (foreground it: the cleanest single experiment in the paper).
Assumes: all of Parts I--V.

**30. `paper-arithmetic` Reading the Paper III: The Arithmetic Information Generalizes** (Sec. IV; Tables IV--V; App. A steps 20, 23). Includes: three levels of supplied information; controls A, B, C1, C2 line by line; the root-ambiguity footnote; the surrogate alignment check and its exploratory status; dephasing; "the obstacle is the parametrization, not the parameter count".
Assumes: all of Parts I--V.

**31. `paper-spurious-floor` Reading the Paper IV: Anatomy of a Spurious Floor** (Sec. II E, V, V A; Fig. 5; App. A steps 9--11). Includes: the 0.232 floor, the null-mode artifact ($c_0\approx2.2\times10^6$), the failed block-coherence mechanism, exact vs surrogate spectra, amplitude control, the general lessons for computational science.
Assumes: all of Parts I--V.

**32. `paper-critique` Reading the Paper V: Discussion, Limits, and How to Go Further** (Sec. VI; App. A; Data Availability). Includes: the proposed falsifiability standard (i)--(iv); the stated limits; an honest list of weak points a referee could press (literature-gap claim rests on a citation-graph keyword audit; one background potential and three bases; operator fits only to zero 1000; Sec. V not rerun in revision; post hoc alignment statistic; heuristic explanation of the failure; "memorization result can be derived" is open); how to reproduce from the Zenodo record; research projects for the reader (graded), e.g. Hermite or neural-network bases, other backgrounds, higher zeros, deriving the memorization result, Dirichlet $L$-functions.
Assumes: all of Parts I--V and Chs. 28--31.

### Appendices (written by the coordinator, not by chapter calls)

A `ap-python` Python for this book. B `ap-notation` Notation and symbol index. C `ap-claims` Map from every claim in the paper to the chapters that explain it. D `ap-numbers` Key numbers of the paper.

---

## 8. Paper reference sheet (R1 numbering; use these numbers)

- Sections: I Introduction (I A Relation to prior work: a Inverse-spectral fits, b Explicit formulae, c Machine learning, d Operator proposals); II Data, models, and protocols (II A Zeros, unfolding; II B Smooth potential and forward solver; II C Hermitian fluctuation fit and out-of-sample protocol, with paragraphs "Other bases", "Within-height splits", "No leakage through the bias correction"; II D Arithmetic oracle, with "Fitted sinusoidal controls"; II E Second-order non-Hermitian classes; II F Use of artificial intelligence); III Fitting is easy, transfer is nil (paragraphs: Other bases, Within-height splits, Permutation and surrogate tests, Interpolation split, Arithmetic-continuation probe); IV The arithmetic information generalizes (Comparable budgets and fitted controls); V Anatomy of a spurious floor (V A Sum rule and locality details); VI Discussion; Appendix A Reconstruction record (steps 1--23).
- Equations: (1) $N_{\rm sm}$; (2) Wu--Sprung $x(V)$; (3) leakage identity; (4) truncated explicit formula $N_{\rm fl}$; (5) second-order shift.
- Figures: 1 out-of-sample prediction, zeros 370--460 (operator corr $-0.07$, six-prime formula $+0.96$); 2 GUE spacing histogram, first $10^4$ zeros; 3 memorization vs $M$ at two scales; 4 interpolation split (fitted potential, correction, predicted spectrum); 5 non-Hermitian exact vs second order, amplitude-controlled floors.
- Tables: I Hermitian fits (12 rows); II other bases; III within-height splits; IV budget comparison (operator, A, B, C1, C2); V oracle by holdout and prime budget.
- The full LaTeX source of the paper is provided to you in context. Quote numbers from it exactly.

---

## 9. Allowed citation keys

Paper and its references: `soniR1` (the paper), `zenodo` (its code and data record), `montgomery`, `odlyzko87`, `berrykeating99`, `schumayer`, `connes`, `bbm`, `bellissard`, `sierra`, `recenthp1` (Yakaboylu), `recenthp2` (Tamburini), `zetatriples`, `wusprung`, `vanzyl`, `slater`, `jaksch`, `francaleclair`, `leclair`, `connes26`, `sdsu`, `henumberfields`, `vanishingorder`, `murmurations`, `theiler`, `schreiber`, `landau`, `gonek`.

Textbooks and classics: `hrw` (Halliday, Resnick, Walker, Fundamentals of Physics), `rhb` (Riley, Hobson, Bence, Mathematical Methods for Physics and Engineering), `griffiths` (Griffiths and Schroeter, Introduction to Quantum Mechanics), `edwards` (H. M. Edwards, Riemann's Zeta Function), `titchmarsh` (Titchmarsh, The Theory of the Riemann Zeta-Function, 2nd ed. revised by Heath-Brown), `davenport` (Davenport, Multiplicative Number Theory), `apostol` (Apostol, Introduction to Analytic Number Theory), `hardywright` (Hardy and Wright, An Introduction to the Theory of Numbers), `mehta` (Mehta, Random Matrices), `gutzwiller` (Gutzwiller, Chaos in Classical and Quantum Mechanics), `bender98` (Bender and Boettcher, PRL 80, 5243, 1998), `nr` (Press et al., Numerical Recipes), `strang` (Strang, Introduction to Linear Algebra), `trefethen` (Trefethen and Bau, Numerical Linear Algebra), `esl` (Hastie, Tibshirani, Friedman, The Elements of Statistical Learning), `efron` (Efron and Tibshirani, An Introduction to the Bootstrap), `marquardt` (Marquardt 1963), `riemann1859` (Riemann's 1859 paper), `brent` (Brent, Algorithms for Minimization without Derivatives).

---

## 10. Running examples (use them, so chapters feel connected)

- **RE1, the first ten zeros**: $t_n$ = 14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005151, 49.773832.
- **RE2, the prime-2 wave**: $-\frac1\pi\sum_{m=1}^3\frac{\sin(mt\log2)}{m2^{m/2}}$, period $2\pi/\log2\approx9.065$ in $t$; the paper's continuation probe target.
- **RE3, the box**: infinite well on $[-L,L]$ with $L=44.883$ (scale 1), then the Wu--Sprung potential in it.
- **RE4, the memorizing polynomial**: a high-degree polynomial fitted to a few noisy points of $\sin t$, used in Chs. 9 and 25 as the simplest picture of memorization.
- **RE5, zeros 401--500**: the paper's principal holdout; null rms 0.263.

When a chapter introduces a tool, it should apply it to at least one running example.

---

## 11. Honesty rules

- Do not overstate the paper. Its claims are scoped: one background potential (Wu--Sprung), three bases, specific optimizer, first 1000 zeros for operator fits, $10^4$ for the oracle. The explanation of the failure is heuristic. Keep the paper's hedges.
- Do not claim novelty for the explicit formula (the paper does not).
- Mark the explicit-formula derivation in Ch. 18 as formal/heuristic where it is; give the rigorous status.
- Never state RH or Hilbert--P\'olya as true.
- The paper was produced with substantial AI assistance (paper Sec. II F); state this neutrally where relevant (Chs. 28, 32).
