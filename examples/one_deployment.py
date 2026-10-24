"""One deployment, three tasks: F_p, 2-join and 3-star from the same sketch.

Run from the artifact root:

    python3 examples/one_deployment.py

This is the paper's "one deployment, three tasks" experiment (Section 5.6).
Three columns are sketched ONCE -- all three share the same bucket hash h and
the same root-of-unity hash g, which is the deployment requirement for a star
-- and the stored counters then answer three different questions without
re-reading the data:

    F_3 of each column      Re( sum_s C[j, s]^3 )
    2-join of two columns   Re( sum_s C^R_s * conj(C^S_s) )   (exponents +1, -1)
    3-star of all three     Re( sum_s C^1_s * C^2_s * C^3_s ) (exponents 1+1+1)

The comparison that matters is against a real-valued toolkit: inside the
power-sum family there is *no* real multiplier that is unbiased for F_3
(the paper's necessity theorem), so covering all three tasks with real
sketches costs an exact key-domain array -- or a non-mergeable AMS sampler.
The script prints both budgets.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ccs  # noqa: E402

U = 1_000_000     # keys in the shared universe
N = 1_000_000     # rows per column
ALPHA = 1.1
P, W, D, SEED = 3, 256, 64, 2024
NCOL = 3


def zipf_column(rng):
    wts = np.arange(1, U + 1, dtype=np.float64) ** (-ALPHA)
    wts /= wts.sum()
    return np.bincount(rng.choice(U, size=N, p=wts), minlength=U).astype(
        np.float64)


def deploy(F):
    """Sketch all columns with ONE shared (h, g) per row; return (CR, CI)."""
    rng = np.random.default_rng(SEED)
    CR = np.empty((NCOL, D, W))
    CI = np.empty((NCOL, D, W))
    for j in range(D):
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        for c, f in enumerate(F):
            CR[c, j] = np.bincount(h, weights=f * gr, minlength=W)
            CI[c, j] = np.bincount(h, weights=f * gi, minlength=W)
    return CR, CI


def pow_rows(cr, ci, p):
    """Per-row Re(sum_s C[j,s]^p) by repeated complex multiplication."""
    re = np.empty(len(cr))
    for j, (a0, b0) in enumerate(zip(cr, ci)):
        a, b = a0.copy(), b0.copy()
        for _ in range(p - 1):
            a, b = a * a0 - b * b0, a * b0 + b * a0
        re[j] = a.sum()
    return re


def main():
    rng = np.random.default_rng(0)
    F = [zipf_column(rng) for _ in range(NCOL)]

    # ---- exact answers, by full scan (this is what the sketch replaces) ----
    truth_fp = [float((f ** P).sum()) for f in F]
    truth_j2 = float((F[0] * F[1]).sum())
    truth_star = float((F[0] * F[1] * F[2]).sum())

    CR, CI = deploy(F)
    bytes_sketch = CR.nbytes + CI.nbytes
    bytes_exact = sum(f.nbytes for f in F)

    print(f"{NCOL} zipf({ALPHA}) columns, |U|={U:,} keys, {N:,} rows each")
    print(f"one deployment: {NCOL} x {D} x {W} complex counters "
          f"= {bytes_sketch / 2**20:.2f} MiB "
          f"(vs {bytes_exact / 2**20:.1f} MiB of exact key-domain arrays, "
          f"{bytes_exact / bytes_sketch:.0f}x)\n")

    print(f"  {'task':<22} {'estimate':>16} {'exact':>16} {'rel. err':>9}")
    for c in range(NCOL):
        est = ccs.median(pow_rows(CR[c], CI[c], P))
        print(f"  {f'F_{P} of column {c + 1}':<22} {est:>16.6e} "
              f"{truth_fp[c]:>16.6e} {abs(est - truth_fp[c]) / truth_fp[c]:>8.3%}")

    est_j2 = ccs.median(ccs.join_rows(CR[0], CI[0], CR[1], CI[1], P))
    print(f"  {'2-join (cols 1,2)':<22} {est_j2:>16.6e} {truth_j2:>16.6e} "
          f"{abs(est_j2 - truth_j2) / truth_j2:>8.3%}")

    est_star = ccs.median(_star_rows(CR, CI))
    print(f"  {'3-star (cols 1,2,3)':<22} {est_star:>16.6e} "
          f"{truth_star:>16.6e} {abs(est_star - truth_star) / truth_star:>8.3%}")

    print("\nAll three answers come from the same counters: the exponents "
          "(+1,-1)\nfor the join and (+1,+1,+1) for the star both sum to 0 "
          "mod p, which is\nwhat makes one stored sketch serve every task.  A "
          "real-valued sketch\ncannot do all three -- there is no unbiased "
          "real multiplier for F_3\ninside the power-sum family -- so it needs "
          "the exact arrays above.")


def _star_rows(CR, CI):
    """Per-row Re of the element-wise product of the stored counters."""
    pr = np.ones((D, W))
    pi = np.zeros((D, W))
    for c in range(NCOL):
        cr, ci = CR[c], CI[c]
        pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
    return pr.sum(axis=1)


if __name__ == "__main__":
    main()
