"""Exp 15 -- five-table stars at order five (paper Section 5.5).

Five round-robin shards of the bundled MovieLens ratings (aligned 5-star,
keys userId / movieId) and the five largest Kasandr categories (misaligned):
CCS order-5 vs byte-matched per-edge real.

Usage: python exp15_five_star.py [families]    (paper: 24; default 12)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import raw, kasandr
import ccs

P, W, DEPTH = 5, 256, 64
FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 12

if __name__ == "__main__":
    ml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "ML", "ratings")
    for key in ("userId", "movieId"):
        arr = raw(f"ML/ratings/{key}.gz").astype(np.int64)
        kcol = np.unique(arr)
        F = []
        for i in range(5):
            k, c = np.unique(arr[i::5], return_counts=True)
            f = np.zeros(len(kcol)); f[np.searchsorted(kcol, k)] = c
            F.append(f)
        truth = float(np.prod(np.vstack([f + 1e-300 for f in F]), axis=0).sum())
        A, E = np.empty(FAMS), np.empty(FAMS)
        for t in range(FAMS):
            rng = np.random.default_rng(9500 + t)
            re, _ = ccs.star_rows(F, P, W, rng, d=DEPTH)
            A[t] = abs(ccs.median(re) - truth) / truth
            rng = np.random.default_rng(9600 + t)
            est = ccs.peredge_star(F, W, rng, d=DEPTH)
            E[t] = abs(ccs.median(est) - truth) / truth
        print(f"[5star ML {key}] truth={truth:.5g} CCS={A.mean():.4g} peredge={E.mean():.4g}",
              flush=True)

    users, cats, times, U = kasandr()
    cc = np.bincount(cats)
    top = np.argsort(cc)[::-1][:5]
    Fs = []
    for cv in top:
        k, c = np.unique(users[cats == cv], return_counts=True)
        f = np.zeros(U); f[k] = c
        Fs.append(f)
    truth = float(np.prod(np.vstack([f + 1e-300 for f in Fs]), axis=0).sum())
    A, E = np.empty(FAMS), np.empty(FAMS)
    for t in range(FAMS):
        rng = np.random.default_rng(9500 + t)
        re, _ = ccs.star_rows(Fs, P, W, rng, d=DEPTH)
        A[t] = abs(ccs.median(re) - truth) / truth
        rng = np.random.default_rng(9600 + t)
        est = ccs.peredge_star(Fs, W, rng, d=DEPTH)
        E[t] = abs(ccs.median(est) - truth) / truth
    print(f"[5star KAS cat] truth={truth:.5g} CCS={A.mean():.4g} peredge={E.mean():.4g}",
          flush=True)
