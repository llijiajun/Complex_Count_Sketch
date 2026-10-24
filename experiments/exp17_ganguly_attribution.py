r"""Experiment 17 -- attributing the Ganguly-vs-CCS gap (paper Section 5.2).

At EQUAL total memory (B*s1 = w*d = 65,536 complex counters, p=3), the root
of unity and the power-sum estimator are the same, so any accuracy difference
must come from (a) the width/depth ALLOCATION and (b) the AGGREGATION.
2x2 design on four synthetic distributions (D=10^4, N=10^7, 10 trials):

  allocation: G-preset (B=100 buckets x 655 copies)  vs  CCS split (1024 x 64)
  aggregation: mean over copies/rows                vs  median over copies/rows

Cells: (G,mean) == the implemented Ganguly baseline; (CCS,median) == CCS.
The other two cells attribute the gap.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

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
    return k, c.astype(np.float64)   # int64 overflows at p=3 for f ~ 4e6

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
            "G-preset/mean":   (100, 655, "mean"),
            "G-preset/median": (100, 655, "median"),
            "CCS-split/mean":  (1024, 64, "mean"),
            "CCS-split/median": (1024, 64, "median"),
        }.items():
            r[name] = run(k, c, B, d, agg, seed0=2000 + 31 * len(name))
        print(f"{dist:9s} " + "  ".join(f"{k2}={v:.3e}" for k2, v in r.items())
              + f"  alloc-only={r['G-preset/median']/r['CCS-split/median']:.1f}x"
              f"  agg-only={r['G-preset/median']/r['G-preset/mean']:.2f}x", flush=True)
    print("EXP17_DONE")
