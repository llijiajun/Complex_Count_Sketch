r"""experiments/exp14_merge_delete.py -- sharded deployment: merge,
communication, and deletions (paper Section 5.6).

Original: paper_code/experiments/exp14_merge_delete.py

MovieLens ratings userId (25M rows) split into 16 shards by uid % 16; every
site builds a CCS order-3 sketch (w=256, d=16) with the SAME shared hash
functions (the deployment requirement), the root merges by counter addition.

Measured:
  (a) merged-vs-centralized estimate gap (exact arithmetic would be 0; float
      addition order gives ~1e-12), and the error of both against truth;
  (b) communication: 16 x d x w complex counters vs the raw stream bytes;
  (c) deletions: every site deletes a uniform 10% of its rows by counter
      subtraction; the merged post-deletion sketch against a centralized
      post-deletion rebuild, both against the post-deletion truth.

Artifact changes vs the original: the MovieLens column is read through
loaders.py (bundled gzip file, byte-identical to the original).  Output
matches results/experiments/exp14_merge_delete.txt.

Usage:
    python3 experiments/exp14_merge_delete.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders

P, W, DROWS, NSITES, T = 3, 256, 16, 16, 10
DEL_FRAC = 0.10


def sketch_row(h, gr, gi, f):
    cr = np.bincount(h, weights=f * gr, minlength=W)
    ci = np.bincount(h, weights=f * gi, minlength=W)
    return cr, ci


def estimate(C):
    re = ((C.real) ** 3 - 3 * C.real * C.imag ** 2).sum(axis=1)
    return float(np.median(re))


if __name__ == "__main__":
    arr = loaders.raw(os.path.join("ML", "ratings", "userId.gz"))
    kcol, counts = np.unique(arr, return_counts=True)
    U = len(kcol)
    # shard membership by uid % 16
    sid = kcol % NSITES
    F = np.zeros((NSITES, U))
    for s in range(NSITES):
        F[s][sid == s] = counts[sid == s]
    Fc = F.sum(axis=0)
    truth = float((Fc ** 3).sum())
    rng = np.random.default_rng(7)
    print(f"# sites={NSITES} rows={arr.size} keys={U} truth(F3)={truth:.6g}")
    max_gap = 0.0
    errs_c, errs_m = [], []
    stream_bytes = arr.size * 5.6            # ~compact ascii per item
    sketch_bytes = NSITES * DROWS * W * 16   # complex double counters
    for t in range(T):
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        Cc = np.zeros((DROWS, W), dtype=complex)   # centralized
        Cm = np.zeros((DROWS, W), dtype=complex)   # merged from sites
        for s in range(NSITES):
            Cs = np.zeros((DROWS, W), dtype=complex)
            for j in range(DROWS):
                cr, ci = sketch_row(h, gr, gi, F[s])
                Cs[j] = cr + 1j * ci
            Cm += Cs
        for j in range(DROWS):
            cr, ci = sketch_row(h, gr, gi, Fc)
            Cc[j] = cr + 1j * ci
        ec = abs(estimate(Cc) - truth) / truth
        em = abs(estimate(Cm) - truth) / truth
        gap = abs(estimate(Cm) - estimate(Cc)) / truth
        errs_c.append(ec); errs_m.append(em); max_gap = max(max_gap, gap)
    print(f"(a) merged-vs-central max est gap={max_gap:.3g}  "
          f"err central={np.mean(errs_c):.4g}  err merged={np.mean(errs_m):.4g}")
    print(f"(b) communication: merged={sketch_bytes/1e6:.2f} MB vs stream={stream_bytes/1e6:.1f} MB "
          f"-> {stream_bytes/sketch_bytes:.0f}x")
    # ---- deletions: each site drops a uniform DEL_FRAC of its rows ----
    keep = (rng.random(arr.size) > DEL_FRAC)
    kcol2, counts2 = np.unique(arr[keep], return_counts=True)
    F2 = np.zeros((NSITES, U))
    pos = {k: i for i, k in enumerate(kcol)}
    idx2 = np.array([pos[k] for k in kcol2])
    np.add.at(F2, (sid[idx2], idx2), counts2)
    Fc2 = F2.sum(axis=0)
    truth2 = float((Fc2 ** 3).sum())
    max_gap2 = 0.0; errs2_c, errs2_m = [], []
    for t in range(T):
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        Cc = np.zeros((DROWS, W), dtype=complex)
        Cm = np.zeros((DROWS, W), dtype=complex)
        for s in range(NSITES):
            Cs = np.zeros((DROWS, W), dtype=complex)
            for j in range(DROWS):
                cr, ci = sketch_row(h, gr, gi, F2[s])
                Cs[j] = cr + 1j * ci
            Cm += Cs
        for j in range(DROWS):
            cr, ci = sketch_row(h, gr, gi, Fc2)
            Cc[j] = cr + 1j * ci
        ec = abs(estimate(Cc) - truth2) / truth2
        em = abs(estimate(Cm) - truth2) / truth2
        errs2_c.append(ec); errs2_m.append(em)
        max_gap2 = max(max_gap2, abs(estimate(Cm) - estimate(Cc)) / truth2)
    print(f"(c) after {DEL_FRAC:.0%} deletions: max est gap={max_gap2:.3g}  "
          f"err central={np.mean(errs2_c):.4g}  err merged={np.mean(errs2_m):.4g}  "
          f"truth(post)={truth2:.6g}")
    print("EXP14_DONE", flush=True)
