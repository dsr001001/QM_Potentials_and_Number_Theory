# Chapter review log

Second review by Claude Fable 5.2 (critique) and Claude Opus 5.5 (revision), one chapter at a time. Earlier rounds are recorded in `work/round1`, `work/round2`.

| Chapter | Issues found (high/med/low) | Patches applied / failed | Checks after revision | Cost (USD) | Finished |
|---|---|---|---|---|---|
| level-statistics | 33 (2/10/21) | 50 / 0 | exit code 0 passed assertions: 136; failures: 0 | 1.61 | 2026-11-14 00:34 |
| sparse-frequency | 31 (1/13/17) | 113 / 0 (1 applied by hand) | exit code 0 passed assertions: 108; failures: 0 | 1.97 | 2026-11-14 00:51 |
| controls | 26 (1/7/18) | 37 / 0 | exit code 0 passed assertions: 86; failures: 0 | 1.49 | 2026-11-14 01:02 |
| least-squares | 28 (1/6/21) | 68 / 0 | exit code 0 passed assertions: 69; failures: 0 | 1.67 | 2026-11-14 01:15 |

## Status after this round (14 Nov 2026)

API credit used up: ledger total $78.54. The queue stopped before finite-differences.

**Second review (Fable 5.2) done, 20 chapters:** intro, wkb, zeta, counting-zeros, explicit-formula, perturbation, non-hermitian, hilbert-polya, wu-sprung, fitting, generalization, all five Part VI chapters (rounds 2 and 3, see `work/round2`), and level-statistics, sparse-frequency, controls, least-squares (this log).

**Single review only (Sonnet 5.5), 12 chapters:** finite-differences, probability, primes, eigen, schrodinger, bound-states, fourier-transform, fourier-series, linear-algebra, asymptotics, series, complex.
| complex | in-session line-by-line read (Claude Opus 5.5, no API): derivations and all quoted numbers rechecked (arg/modulus of rho_1, rho_2, 2^{-rho_1}, RE2 bound 0.0487 vs observed 0.042, t_400); no errors; 1 garbled answer (25c) rewritten, 1 'one checks' gap expanded | 2 edits by hand | exit code 0 passed assertions: 203; failures: 0; standalone compile: 0 errors | 0 (plan) | 2026-11-14 02:05 |
