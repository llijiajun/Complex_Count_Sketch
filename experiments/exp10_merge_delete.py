"""Exp 10 -- sharded deployment: merge, communication, deletions (Sec 5.6).

16 sites shard the bundled MovieLens ratings userId by uid % 16; each builds
an order-3 CCS (w=256, d=16) with the SHARED hash family; the root merges by
counter addition.  Verifies merged-vs-centralized agreement, communication,
and deletions by counter subtraction.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import raw

P, W, DEPTH, NSITES, T, DEL = 3, 256, 16, 16, 10, 0.10

if __name__ == "__main__":
    arr = raw("ML/ratings/userId.gz").astype(np.int64)
    kcol, counts = np.unique(arr, return_counts=True)
    U = len(kcol)
    sid = kcol % NSITES
    F = np.zeros((NSITES, U))
    for s in range(NSITES):
        F[s][sid == s] = counts[sid == s]
    Fc = F.sum(axis=0)
    truth = float((Fc ** P).sum())
    rng = np.random.default_rng(7)
    print(f"sites={NSITES} rows={arr.size} keys={U} truth(F3)={truth:.6g}")

    def rows_of(f, h, gr, gi):
        CR = np.empty((DEPTH, W)); CI = np.empty((DEPTH, W))
        for j in range(DEPTH):
            CR[j] = np.bincount(h, weights=f * gr, minlength=W)
            CI[j] = np.bincount(h, weights=f * gi, minlength=W)
        return CR, CI

    def estimate(C):
        X = C ** P
        return float(np.median(X.real.sum(axis=1)))

    def build(Fmat):
        Cm = np.zeros((DEPTH, W), dtype=complex)
        for s in range(NSITES):
            Cm += build_one(Fmat[s], h, gr, gi)
        return Cm

    def build_one(f, h, gr, gi):
        CR, CI = rows_of(f, h, gr, gi)
        return CR + 1j * CI

    max_gap, errs_m = 0.0, []
    for t in range(T):
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        Cc = build_one(Fc, h, gr, gi)
        Cm = np.zeros_like(Cc)
        for s in range(NSITES):
            Cm = Cm + build_one(F[s], h, gr, gi)
        ec = abs(estimate(Cc) - truth) / truth
        em = abs(estimate(Cm) - truth) / truth
        errs_m.append(em); max_gap = max(max_gap, abs(estimate(Cm) - estimate(Cc)) / truth)
    print(f"merged-vs-central max est gap = {max_gap:.3g}; err merged = {np.mean(errs_m):.4g}")
    sb = NSITES * DEPTH * W * 16
    print(f"communication: merged = {sb/1e6:.2f} MB vs stream = {arr.size*5.6/1e6:.1f} MB "
          f"-> {arr.size*5.6/sb:.0f}x")

    # deletions
    keep = rng.random(arr.size) > DEL
    k2, c2 = np.unique(arr[keep], return_counts=True)
    F2 = np.zeros((NSITES, U))
    Fc2 = np.zeros(U)
    pos = np.searchsorted(kcol, k2)
    for s in range(NSITES):
        m = sid[pos] == s
        np.add.at(F2[s], pos[m], c2[m])
        np.add.at(Fc2, pos[m], c2[m])
    truth2 = float((Fc2 ** P).sum())
    max_gap2, errs2 = 0.0, []
    for t in range(T):
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        Cc = build_one(Fc2, h, gr, gi)
        Cm = np.zeros_like(Cc)
        for s in range(NSITES):
            Cm = Cm + build_one(F2[s], h, gr, gi)
        e = abs(estimate(Cm) - truth2) / truth2
        errs2.append(e); max_gap2 = max(max_gap2, abs(estimate(Cm) - estimate(Cc)) / truth2)
    print(f"after {DEL:.0%} deletions: max est gap = {max_gap2:.3g}; "
          f"err merged = {np.mean(errs2):.4g} (post-deletion truth)")
    print("EXP-MERGE DONE")
