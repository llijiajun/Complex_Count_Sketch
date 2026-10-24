r"""experiments/exp16_real_solutions.py -- exact real multipliers at odd p
(paper Section 4.2).

Original: paper_code/experiments/exp16_real_solutions.py

Real multipliers are obtained by EXACT linear solve on small fixed supports:
  p=3: {-1,1,2}          w = {7/12, 1/4, 1/6}   (the paper's solution)
  p=5: {-1,0.5,1.5,2.5}  (exact 4x4 solve)
  p=7: {-3.5,-3,-2,2.5,3.5} (exact 5x5 solve; chosen for the smallest
        measured variance ratio among feasible supports)
Feasibility note: the previously circulated support {-2.5,-1,-0.25,1,2,3}
admits NO solution (max achievable E[s^7] under the constraints is 0.94<1).

Verification:
  (a) odd moments and mu_p exact to ~1e-15;
  (b) unbiasedness by EXACT expectation: E[(sum_alpha f_alpha S_alpha)^p]
      computed by dynamic programming over the multiplier moments, compared
      with F_p (machine precision; no Monte Carlo);
  (c) variance ratio of the real solution against the root-of-unity CCS
      multiplier on identical streams (Section 4.2's quoted number).

Artifact changes vs the original: the exact solve / moment / expectation
utilities now live in ccs.py (exact_solve_real_multiplier / g_moments /
exact_power_expectation -- identical mathematics; the original's dead
fallback loop inside exact_expectation, which called the removed
np.math.factorial, is dropped).  Output matches
results/experiments/exp16_real_solutions.txt.

Usage:
    python3 experiments/exp16_real_solutions.py
"""
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs


def exact_solve(p, sup):
    return ccs.exact_solve_real_multiplier(p, np.asarray(sup, dtype=float))


def moments(p, s, w):
    worst = max(abs(float((w * s ** m).sum())) for m in range(1, p, 2))
    return worst, float((w * s ** p).sum())


def variance_ratio(p, s, w, families=30, w64=64, d=64, D=200, N=50_000, seed=11):
    rng0 = np.random.default_rng(seed)
    freq = np.bincount(rng0.integers(0, D, size=N), minlength=D).astype(np.float64)
    vr = []
    for t in range(families):
        rng = np.random.default_rng(1000 * p + t)
        rows = np.empty(d)
        for j in range(d):
            h = rng.integers(0, w64, size=D)
            k = rng.choice(len(s), size=D, p=w)
            C = np.bincount(h, weights=freq * s[k], minlength=w64)
            rows[j] = float((C ** p).sum())
        rng2 = np.random.default_rng(9000 * p + t)
        rows_c = np.empty(d)
        for j in range(d):
            h = rng2.integers(0, w64, size=D)
            k = rng2.integers(0, p, size=D)
            g = np.exp(2j * np.pi * k / p)
            Cr = np.bincount(h, weights=freq * g.real, minlength=w64)
            Ci = np.bincount(h, weights=freq * g.imag, minlength=w64)
            rows_c[j] = float(((Cr + 1j * Ci) ** p).real.sum())
        vr.append(np.var(rows) / np.var(rows_c))
    return float(np.median(vr))


if __name__ == "__main__":
    sup3 = np.array([-1.0, 1.0, 2.0]); w3 = np.array([7 / 12, 1 / 4, 1 / 6])
    sup5 = np.array([-1.0, 0.5, 1.5, 2.5]); w5 = exact_solve(5, sup5)
    # p=7: pick the mildest feasible support among a small search
    pool = [-3.5, -3, -2.5, -2, -1.5, -1, 1.5, 2, 2.5, 3, 3.5, 4]
    best = None
    for sup in combinations(pool, 5):
        s7 = np.array(sup, dtype=float)
        try:
            w7 = exact_solve(7, s7)
        except AssertionError:
            continue
        r = variance_ratio(7, s7, w7, families=8)
        if best is None or r < best[0]:
            best = (r, s7, w7)
    sup7, w7 = best[1], best[2]
    print(f"p=7 mildest support found: {sup7.tolist()} (quick ratio {best[0]:.1f})")

    for p, s, w in ((3, sup3, w3), (5, sup5, w5), (7, sup7, w7)):
        worst, mu_p = moments(p, s, w)
        mu = np.array([float((w * s ** m).sum()) for m in range(p + 1)])
        mu[0] = 1.0
        freq = np.bincount(np.random.default_rng(11).integers(0, 200, size=50_000),
                           minlength=200).astype(np.float64)
        E = ccs.exact_power_expectation(p, freq, mu)
        truth = float((freq ** p).sum())
        ratio = variance_ratio(p, s, w)
        print(f"p={p}: support={s.tolist()}")
        print(f"   weights={[round(float(x), 8) for x in w.tolist()]}")
        print(f"   worst |odd moment| = {worst:.2e}   mu_p - 1 = {mu_p - 1:.2e}")
        print(f"   exact E[sum C^p] / F_p - 1 = {E / truth - 1:.2e}")
        print(f"   variance ratio real/CCS = {ratio:.1f}")
    print("EXP16_DONE")
