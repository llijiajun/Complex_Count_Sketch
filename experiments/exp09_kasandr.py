"""Exp 09 -- real Kasandr 3-stars (paper Section 5.5).

(a) Aligned: three time-window shards of the 15.8M-row train set.
(b) Misaligned: the three largest offer categories as verticals.
CCS p=3 vs byte-matched per-edge real vs shared-sign.

Usage: python exp09_kasandr.py [families]    (paper: 24; default 12)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import kasandr
import ccs

P, W, DEPTH = 3, 256, 64
FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 12

if __name__ == "__main__":
    users, cats, times, U = kasandr()
    print(f"Kasandr: {len(users)} rows, {U} users", flush=True)

    def frag(mask):
        k, c = np.unique(users[mask], return_counts=True)
        f = np.zeros(U); f[k] = c
        return f

    def run(tag, F):
        truth = float((F[0] * F[1] * F[2]).sum())
        A, E, S, NEG = np.empty(FAMS), np.empty(FAMS), np.empty(FAMS), 0
        for t in range(FAMS):
            rng = np.random.default_rng(9000 + t)
            re, im = ccs.star_rows(F, P, W, rng, d=DEPTH)
            A[t] = abs(ccs.median(re) - truth) / truth
            rng = np.random.default_rng(9100 + t)
            est = ccs.peredge_star(F, W, rng, d=DEPTH)
            E[t] = abs(ccs.median(est) - truth) / truth
            rng = np.random.default_rng(9200 + t)
            est = ccs.sharedsign_star(F, W, rng, d=DEPTH)
            m = ccs.median(est); NEG += (m < 0)
            S[t] = abs(m - truth) / truth
        print(f"[{tag}] truth={truth:.5g} CCS={A.mean():.4g} "
              f"peredge={E.mean():.4g} sharedsign={S.mean():.4g} neg={NEG}/{FAMS}",
              flush=True)

    # (a) aligned: three time-window shards
    q1, q2 = np.percentile(times, [100 / 3, 200 / 3])
    masks = [times <= q1, (times > q1) & (times <= q2), times > q2]
    run("aligned time-windows", [frag(m) for m in masks])

    # (b) misaligned: top-3 categories
    cc = np.bincount(cats)
    top = np.argsort(cc)[::-1][:3]
    run("misaligned top-3 categories", [frag(cats == c) for c in top])
