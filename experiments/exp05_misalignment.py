"""Exp 05 -- misalignment vs skew at fixed marginals (paper Section 5.4).

Three columns over D=2000 keys share the SAME frequency multiset (truncated
Zipf); a fraction L of each column's top-40 heavy masses is displaced onto
random light keys.  Marginals are identical at every level: any error growth
in L is misplacement, not skew.  Also computes the exact pairwise measure
Psi = sum_{a<b} Mix(a,b)^2 / (2 J^2) from the true frequencies.

Usage: python exp05_misalignment.py [families]   (paper: 60; default 30)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

D, N, H = 2000, 2_000_000, 40
W, DEPTH, P = 256, 64, 3
FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 30


def make_cols(alpha, L, rng):
    w = np.arange(1, D + 1, dtype=np.float64) ** (-alpha)
    f = (w / w.sum() * N).round()
    heavy = np.arange(H); light = np.arange(H, D)
    cols = [f.copy()]
    for _ in range(2):
        g = f.copy()
        move = rng.random(H) < L
        dst = rng.choice(light, size=int(move.sum()))
        np.add.at(g, dst, g[heavy[move]])
        g[heavy[move]] = 0.0
        cols.append(g)
    return cols


def psi_of(cols):
    fa = np.stack(cols); pa = fa.prod(axis=0)
    PiSum = (fa[:, None, :] + fa[:, :, None]).prod(axis=0)
    Mix = PiSum - np.outer(pa, np.ones(len(pa))) - np.outer(np.ones(len(pa)), pa)
    np.fill_diagonal(Mix, 0.0)
    J = float(pa.sum())
    return float((Mix ** 2).sum() / 2 / (2 * J ** 2)), J


if __name__ == "__main__":
    print("# alpha, L, mean rel err, exact Psi, J")
    for alpha in (0.8, 1.5):
        for L in (0.0, 0.25, 0.5, 0.75, 1.0):
            cols = make_cols(alpha, L, np.random.default_rng(424242))
            psi, J = psi_of(cols)
            errs = np.empty(FAMS)
            for t in range(FAMS):
                rng = np.random.default_rng(1000 + int(100 * alpha + 10 * L) + t)
                re, _ = ccs.star_rows(cols, P, W, rng, d=DEPTH)
                errs[t] = abs(ccs.median(re) - J) / J
            print(f"alpha={alpha:.1f} L={L:.2f} err={errs.mean():.4g} psi={psi:.4g} J={J:.5g}",
                  flush=True)
