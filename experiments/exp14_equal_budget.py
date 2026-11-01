"""Exp 14 -- CCS at AMS's own budget (paper Section 5.2, equal-budget rows).

CCS contracted to w=100, d=6 (AMS's 7.2KB at eps=0.1, delta=0.01) on the
bundled Lp synthetic columns.  The paper's headline: on skewed data CCS
stays ahead of AMS at every order; on uniform the contracted sketch
collapses at p>=4 (AMS columns archived from the C++ pipeline).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column
import ccs

P_ORDERS = (3, 4, 5)
W, DEPTH, T = 100, 6, 30

if __name__ == "__main__":
    for name in ("uniform", "zipf1.1", "zipf1.5"):
        k, c = lp_column(f"{name}.dat")
        f = np.zeros(len(k)); f[np.searchsorted(k, k)] = c
        line = [f"{name:9s}"]
        for p in P_ORDERS:
            truth = float((f ** p).sum())
            errs = np.empty(T)
            for t in range(T):
                rng = np.random.default_rng(100 * p + t)
                re, _ = ccs.f_p_rows(f, p, W, rng, d=DEPTH)
                errs[t] = abs(float(np.median(re)) - truth) / truth
            line.append(f"p={p}: {errs.mean():.3e}")
        print("  ".join(line), flush=True)
