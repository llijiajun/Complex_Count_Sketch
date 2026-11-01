"""Exp 02 -- T-MCCS vs plain median vs controls (paper Table 3).

Protocol of Table 3: D=1000, N=1e5, w=256, d=64, p=3,4,5, Zipf skewness
alpha in {0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0}.  Four aggregators of the SAME
sketches: plain median, T-MCCS (Tukey on |Im|), Fixed20 (always drop the 20%
rows with the largest |Im|), Rand20 (drop 20% uniformly at random).

Usage: python exp02_tmccs_table.py [trials]      (paper: 1000; default 200)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

D, N, W, DEPTH = 1000, 100_000, 256, 64
TRIALS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 200


def make_freq(alpha, seed=12345):
    rng = np.random.default_rng(seed)
    if alpha == 0.0:
        items = rng.integers(0, D, size=N)
    else:
        w = np.arange(1, D + 1, dtype=np.float64) ** (-alpha)
        items = rng.choice(D, size=N, p=w / w.sum())
    return np.bincount(items, minlength=D).astype(np.float64)


def fixed20(re, im):
    return float(np.median(re[np.argsort(im)[: int(round(DEPTH * 0.8))]]))


def rand20(re, rng):
    drop = rng.choice(DEPTH, size=int(round(DEPTH * 0.2)), replace=False)
    keep = np.setdiff1d(np.arange(DEPTH), drop)
    return float(np.median(re[keep]))


if __name__ == "__main__":
    for p in (3, 4, 5):
        print(f"--- p={p} (absolute relative error, {TRIALS} trials) ---")
        for alpha in (0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0):
            freq = make_freq(alpha)
            truth = float((freq ** p).sum())
            A = np.empty(TRIALS); G = np.empty(TRIALS)
            B = np.empty(TRIALS); R = np.empty(TRIALS)
            for t in range(TRIALS):
                rng = np.random.default_rng(1000 + t)
                re, im = ccs.f_p_rows(freq, p, W, rng, d=DEPTH)
                A[t] = abs(ccs.median(re) - truth) / truth
                G[t] = abs(ccs.tmccs(re, im) - truth) / truth
                B[t] = abs(fixed20(re, im) - truth) / truth
                R[t] = abs(rand20(re, np.random.default_rng(5000 + t)) - truth) / truth
            wins = int((G < A).sum())
            print(f"alpha={alpha:.1f}: median={A.mean():.4e} tmccs={G.mean():.4e} "
                  f"fixed20={B.mean():.4e} rand20={R.mean():.4e} "
                  f"T/med={G.mean()/A.mean():.3f} wins={wins}/{TRIALS}", flush=True)
