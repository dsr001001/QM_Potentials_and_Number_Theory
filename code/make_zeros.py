"""Compute imaginary parts of the first 10000 nontrivial zeta zeros with mpmath.
First 1000 at 20 significant digits, rest at 15 (as in the paper, App. A step 1)."""
import mpmath, numpy as np, sys, time
from multiprocessing import Pool
def one(n):
    mpmath.mp.dps = 25 if n <= 1000 else 20
    return n, mpmath.nstr(mpmath.zetazero(n).imag, 20 if n <= 1000 else 15)
if __name__ == "__main__":
    nmax = int(sys.argv[1]); t0 = time.time()
    with Pool(2) as p:
        res = sorted(p.map(one, range(1, nmax + 1), chunksize=50))
    with open("qmnt/data/zeros.txt", "w") as f:
        for n, s in res: f.write(f"{n} {s}\n")
    np.save("qmnt/data/zeros.npy", np.array([float(s) for _, s in res]))
    print("done", nmax, time.time() - t0)
