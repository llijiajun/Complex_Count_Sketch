"""Exp 03 -- row-aggregator comparison (paper Section 5.3).

Four aggregators of the SAME sketches under the Table 3 protocol: plain
median, arithmetic mean, median-of-means (8 groups of 8), T-MCCS.

Usage: python exp03_aggregators.py [trials]     (paper: 1000; default 200)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
from exp02_tmccs_table import make_freq, TRIALS

D, N, W, DEPTH = 1000, 100_000, 256, 64

if __name__ == "__main__":
    for p in (3, 4, 5):
        print(f"--- p={p} ---")
        for alpha in (0.0, 0.8, 1.1, 1.5, 2.0):
            freq = make_freq(alpha)
            truth = float((freq ** p).sum())
            med = np.empty(TRIALS); mn = np.empty(TRIALS)
            mm = np.empty(TRIALS); tm = np.empty(TRIALS)
            for t in range(TRIALS):
                rng = np.random.default_rng(1000 + t)
                re, im = ccs.f_p_rows(freq, p, W, rng, d=DEPTH)
                med[t] = abs(ccs.median(re) - truth) / truth
                mn[t] = abs(ccs.mean(re) - truth) / truth
                mm[t] = abs(ccs.med_of_means(re) - truth) / truth
                tm[t] = abs(ccs.tmccs(re, im) - truth) / truth
            print(f"alpha={alpha:.1f}: median={med.mean():.4e} mean={mn.mean():.4e} "
                  f"medofmeans={mm.mean():.4e} tmccs={tm.mean():.4e} "
                  f"mean/med={mn.mean()/med.mean():.2f} "
                  f"MoM/med={mm.mean()/med.mean():.2f} "
                  f"T/med={tm.mean()/med.mean():.3f}", flush=True)
