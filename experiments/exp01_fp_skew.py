"""Exp 01 -- F_p estimation on synthetic data (paper Table 2, CCS columns;
width sweep; 95% CIs; the equal-byte sweep behind Sec. 5.2).

CCS order-p sketches over the bundled Lp columns (uniform / zipf1.1 /
zipf1.5, D=10^4, N=10^7).  The AMS columns of Table 2 were produced by the
original C++ pipeline and are shipped in results/ as reference.

Usage: python exp01_fp_skew.py [n_families]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column
import ccs

FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 10   # paper Table 2: 10

def rel(err, truth):
    return abs(err - truth) / truth

if __name__ == "__main__":
    print("=== F_p on synthetic data (CCS columns of Table 2) ===")
    for name in ("uniform", "zipf1.1", "zipf1.5"):
        k, c = lp_column(f"{name}.dat")
        f = np.zeros(len(k)); f[np.searchsorted(k, k)] = c
        for p in (3, 4, 5):
            truth = float((f ** p).sum())
            errs = np.empty(FAMS)
            for t in range(FAMS):
                rng = np.random.default_rng(100 * p + t)
                re, _ = ccs.f_p_rows(f, p, 256, rng, d=6)   # eps=0.1, d=6
                errs[t] = rel(float(np.median(re)), truth)
            lo, hi = np.percentile(errs, [2.5, 97.5])
            print(f"{name:9s} p={p}: mean={errs.mean():.3e}  "
                  f"95% CI=[{lo:.2e},{hi:.2e}]")

    print("=== Width sweep (Figure 2; zipf1.1, d=64) ===")
    k, c = lp_column("zipf1.1.dat")
    f = np.zeros(len(k)); f[np.searchsorted(k, k)] = c
    p = 3
    truth = float((f ** p).sum())
    for w in (128, 256, 512, 1024, 2048):
        errs = np.empty(FAMS)
        for t in range(FAMS):
            rng = np.random.default_rng(500 + t)
            re, _ = ccs.f_p_rows(f, p, w, rng, d=64)
            errs[t] = rel(float(np.median(re)), truth)
        print(f"p={p} w={w}: mean relative error = {errs.mean():.3e}")
