"""Exp 12 -- Ganguly-vs-CCS gap attribution (paper Section 5.2).

At EQUAL memory (B*s1 = w*d = 65,536 complex counters, p=3) the hash and the
power-sum form are identical, so a 2x2 design over ALLOCATION (Ganguly's
B=100 x 655 copies vs CCS's 1024 x 64 rows) and AGGREGATION (mean vs median)
attributes the accuracy difference.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column

D, N, P = 10_000, 10_000_000, 3

def gen(dist, seed=7):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        a = rng.integers(0, D, size=N)
    elif dist == "mix":
        a1 = rng.integers(0, D, size=N // 2)
        wts = np.arange(1, D + 1) ** (-1.1)
        a2 = rng.choice(D, size=N - N // 2, p=wts / wts.sum())
        a = np.concatenate([a1, a2])
    else:
        wts = np.arange(1, D + 1) ** (-float(dist[4:]))
        a = rng.choice(D, size=N, p=wts / wts.sum())
    k, c = np.unique(a, return_counts=True)
    return k, c.astype(np.float64)          # float: int64 overflows at p=3

def run(k, c, B, d, agg, seed0, trials=10):
    U = len(k)
    errs = np.empty(trials)
    for t in range(trials):
        rng = np.random.default_rng(seed0 + t)
        rows = np.empty(d)
        for j in range(d):
            e = rng.integers(0, P, size=U)
            ang = (2.0 * np.pi / P) * e
            h = rng.integers(0, B, size=U)
            zr = np.bincount(h, weights=c * np.cos(ang), minlength=B)
            zi = np.bincount(h, weights=c * np.sin(ang), minlength=B)
            pr, pi = zr.copy(), zi.copy()
            for _ in range(P - 1):
                pr, pi = pr * zr - pi * zi, pr * zi + pi * zr
            rows[j] = pr.sum()
        est = rows.mean() if agg == "mean" else float(np.median(rows))
        errs[t] = abs(est - truth) / truth
    return errs.mean()

if __name__ == "__main__":
    for dist in ("zipf1.1", "zipf1.5", "uniform", "mix"):
        k, c = gen(dist)
        truth = float((c ** P).sum())
        r = {}
        for name, (B, d, agg) in {
            "G-preset/mean":    (100, 655, "mean"),
            "G-preset/median":  (100, 655, "median"),
            "CCS-split/mean":   (1024, 64, "mean"),
            "CCS-split/median": (1024, 64, "median"),
        }.items():
            r[name] = run(k, c, B, d, agg, seed0=2000 + 31 * len(name))
        print(f"{dist:9s} " + "  ".join(f"{k2}={v:.3e}" for k2, v in r.items())
              + f"  alloc-only={r['G-preset/median']/r['CCS-split/median']:.1f}x"
              f"  agg-only={r['G-preset/median']/r['G-preset/mean']:.2f}x", flush=True)
