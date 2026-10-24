r"""experiments/exp15_planner_rank.py -- from estimates to planning: ranking
columns by F_3 (paper Section 5.7).

Original: paper_code/experiments/exp15_planner_rank.py

Planners order relations by statistics of their key columns; we measure how
well F_3 estimates recover the TRUE F_3 ranking of a dataset's columns.

Datasets: Airlines (10 columns), Kasandr (7 tab-separated fields).
Methods, same sketches and seeds per dataset:
  CCS      order-3 shared-root sketch (w=256, d=64), mean of the median
           estimate over 24 hash families;
  +-1 cubes real power-sum of a shared-sign Count Sketch (biased);
  sample   1% uniform sample, scaled empirical F_3 (the planner's sampling
           baseline).
Metrics: Spearman rank correlation with the true F_3 across columns, and
top-3 recall (does the method select the true three largest columns?).

Artifact changes vs the original:
  * column loading goes through loaders.py (bundled gzip columns);
  * the Kasandr string fields (offerid / countrycode / merchant) were keyed
    with Python's built-in hash(), which is salted per process
    (PYTHONHASHSEED) and hence NOT reproducible; they are now keyed with
    zlib.crc32 (see loaders.kasandr_extra).  The true F_3 order, the CCS and
    the cubes estimates depend only on the distinct-key multiplicities and
    are unchanged; only the sampling baseline's key order differs (its rank
    metrics are unaffected at these magnitude gaps).
Output matches results/experiments/exp15_planner_rank.txt.

Usage:
    python3 experiments/exp15_planner_rank.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders

P, W, DROWS, T = 3, 256, 64, 24


def est_ccs(f, rng0):
    U = len(f)
    est = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(rng0 + t)
        re = np.empty(DROWS)
        for j in range(DROWS):
            h = rng.integers(0, W, size=U)
            g = rng.integers(0, P, size=U)
            ang = (2.0 * np.pi / P) * g
            gr, gi = np.cos(ang), np.sin(ang)
            cr = np.bincount(h, weights=f * gr, minlength=W)
            ci = np.bincount(h, weights=f * gi, minlength=W)
            X = (cr + 1j * ci) ** P
            re[j] = X.real.sum()
        est[t] = float(np.median(re))
    return float(est.mean())


def est_cubes(f, rng0):
    U = len(f)
    est = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(rng0 + t)
        vals = np.empty(DROWS)
        for j in range(DROWS):
            h = rng.integers(0, W, size=U)
            s = rng.integers(0, 2, size=U) * 2 - 1
            c = np.bincount(h, weights=f * s, minlength=W)
            vals[j] = float((c ** 3).sum())
        est[t] = float(np.median(vals))
    return float(est.mean())


def est_sample(k, c, rng0):
    """1% uniform sample over raw rows -> scaled empirical F_3."""
    n = int(c.sum())
    take = max(1, int(0.01 * n))
    rng = np.random.default_rng(rng0)
    p = c / c.sum()
    idx = rng.choice(len(k), size=take, p=p)
    cnt = np.bincount(idx, minlength=len(k))
    return float(((cnt * (n / take)) ** 3).sum())


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_recall(est, true, k=3):
    return len(set(np.argsort(est)[::-1][:k]) & set(np.argsort(true)[::-1][:k])) / k


def run_dataset(name, cols, seed_ccs=5000, seed_cubes=6000, seed_sample=7000):
    """cols: list of (distinct keys, counts) or raw row arrays."""
    freqs = []
    for i, v in enumerate(cols):
        if isinstance(v, tuple):
            k, c = v
        else:
            k, c = np.unique(v, return_counts=True)
        f = np.zeros(len(k), dtype=np.float64)
        f[np.searchsorted(k, k)] = c
        freqs.append((i, f, k, c))
    names = [str(i) for i, _, _, _ in freqs]
    trueF = [float((f ** 3).sum()) for _, f, _, _ in freqs]
    ccse, cube, samp = [], [], []
    for m, (_, f, k, c) in enumerate(freqs):
        ccse.append(est_ccs(f, seed_ccs + 97 * m))
        cube.append(est_cubes(f, seed_cubes + 89 * m))
        samp.append(est_sample(k, c, seed_sample + 83 * m))
    print(f"[{name}] columns={names}")
    print(f"  true F3 order: {np.argsort(trueF)[::-1].tolist()}")
    for meth, est in (("CCS", ccse), ("cubes", cube), ("sample", samp)):
        print(f"  {meth:7s} spearman={spearman(est, trueF):.3f} top3={topk_recall(est, trueF):.2f}",
              flush=True)


if __name__ == "__main__":
    run_dataset("Airlines", [loaders.airlines(c) for c in loaders.AIRLINES])
    users, cats, times, U = loaders.kasandr()
    kas_cols = [users, cats, times,
                loaders.kasandr_extra("offerid"),
                loaders.kasandr_extra("countrycode"),
                loaders.kasandr_extra("merchant")]
    run_dataset("Kasandr", kas_cols, seed_ccs=8000, seed_cubes=9000,
                seed_sample=9500)
    print("EXP15_DONE", flush=True)
