"""Exp 04 -- p-star join: CCS vs real-valued baselines (paper Table 4).

Six synthetic 3-column combinations (independently regenerated columns from
the bundled Lp files), p=3, w=256, d=64.  Estimators: CCS (shared root),
T-MCCS (same sketches), per-edge Rademacher (unbiased, byte-matched width
2w), shared-sign real product (biased), cubes of one real sketch (biased).

Usage: python exp04_pstar_baselines.py [families]   (paper: 100; default 30)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column, merge_universe
import ccs

P, W, DEPTH = 3, 256, 64
FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 30
SUF = ["", "2", "3"]
COMBOS = [("uniform", "uniform", "uniform"),
          ("uniform", "uniform", "zipf1.1"),
          ("uniform", "zipf1.1", "zipf1.5"),
          ("uniform", "zipf1.5", "zipf1.5"),
          ("zipf1.1", "zipf1.1", "zipf1.1"),
          ("zipf1.5", "zipf1.5", "zipf1.5")]

if __name__ == "__main__":
    for names in COMBOS:
        fc = [lp_column(t + ".dat" + s) for t, s in zip(names, SUFS)]
        allk, F = merge_universe(fc)
        common = fc[0][0]
        for k, _ in fc[1:]:
            common = np.intersect1d(common, k, assume_unique=True)
        truth = np.ones(len(common))
        for k, c in fc:
            truth *= c[np.searchsorted(k, common)]
        truth = float(truth.sum())

        A = np.empty(FAMS); G = np.empty(FAMS)
        E = np.empty(FAMS); S = np.empty(FAMS); NEG = 0
        for t in range(FAMS):
            rng = np.random.default_rng(7000 + t)
            re, im = ccs.star_rows(F, P, W, rng, d=DEPTH)
            A[t] = abs(ccs.median(re) - truth) / truth
            G[t] = abs(ccs.tmccs(re, im) - truth) / truth
            rng = np.random.default_rng(8100 + t)
            est = ccs.peredge_star(F, W, rng, d=DEPTH)
            E[t] = abs(ccs.median(est) - truth) / truth
            rng = np.random.default_rng(8200 + t)
            est = ccs.sharedsign_star(F, W, rng, d=DEPTH)
            m = ccs.median(est); NEG += (m < 0)
            S[t] = abs(m - truth) / truth
        print(f"{'x'.join(n[:6] for n in names):20s} truth={truth:.5g} "
              f"CCS={A.mean():.4g} T-MCCS={G.mean():.4g} G/A={G.mean()/A.mean():.2f} "
              f"peredge={E.mean():.4g} sharedsign={S.mean():.4g} neg={NEG}/{FAMS}",
              flush=True)
