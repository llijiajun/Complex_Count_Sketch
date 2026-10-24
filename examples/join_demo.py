"""Join-size estimation with CCS: multi-way (p-star) and 2-way joins.

Run from the artifact root:

    python3 examples/join_demo.py

Two worked examples, both comparing against the exact answer:

1. A 3-way "star" join on a shared key.  The join size is
   sum_a f1_a * f2_a * f3_a.  CCS estimates it at every order p >= 3 using
   complex roots of unity; the natural real-valued alternatives are shown
   alongside (per-edge Rademacher, shared-sign, and "cube the merged
   stream").
2. A 2-way join, where one order-p sketch per relation answers |R join S| by
   Re(sum_s C^R_s * conj(C^S_s)); the classical Count Sketch is the
   order-2 specialist baseline.

Read the numbers honestly, as the paper does: at equal bytes the per-edge
real sketch is usually the *stronger* star estimator of the two.  What the
shared complex root buys instead is (i) one hash per key rather than p-1,
(ii) the exact collision law behind the misalignment diagnosis, (iii) the
imaginary-part signal that T-MCCS trades on and a real sketch does not have,
and (iv) one deployment that also answers F_p -- see one_deployment.py.  The
biased real constructions (shared sign, cubes) fail in a way CCS does not:
their median estimate is outright negative in about half the hash families.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ccs  # noqa: E402

U = 200_000        # keys in the shared universe
ALPHA = 1.1        # skew of the key distribution
WIDTH = 256        # buckets per row
DEPTH = 64         # rows
T = 10             # independent hash families averaged over


def rel(est, truth):
    return abs(est - truth) / truth


def star_demo():
    """3-way star join: CCS vs the real-valued baselines."""
    rng = np.random.default_rng(0)
    wts = np.arange(1, U + 1, dtype=np.float64) ** (-ALPHA)
    wts /= wts.sum()
    F = [np.bincount(rng.choice(U, size=400_000, p=wts), minlength=U)
         .astype(np.float64) for _ in range(3)]
    truth = float((F[0] * F[1] * F[2]).sum())

    print(f"3-way star join: |U|={U:,} keys, 3 x 400k rows, Zipf({ALPHA})")
    print(f"exact join size = {truth:.6e}\n")

    def over_families(fn):
        """Mean/median relative error of `fn(seed)` over T hash families."""
        errs = np.empty(T)
        for t in range(T):
            errs[t] = rel(fn(1000 + t), truth)
        return float(errs.mean()), float(np.median(errs))

    # --- CCS: complex roots of unity, one shared (h, g) across all columns ---
    ccs_m, _ = over_families(
        lambda s: ccs.star_estimate(F, p=3, width=WIDTH, seed=s, depth=DEPTH))
    ccs_t, _ = over_families(
        lambda s: ccs.star_estimate(F, p=3, width=WIDTH, seed=s, depth=DEPTH,
                                    agg="tmccs"))

    # --- per-edge Rademacher: unbiased, but one independent +-1 per edge. ---
    # Called with the same `width`, i.e. 2*width real counters per column,
    # which is byte-for-byte the same memory as width complex counters.
    per_m, _ = over_families(
        lambda s: ccs.median(ccs.peredge_star(F, WIDTH,
                                              np.random.default_rng(s),
                                              d=DEPTH)))

    # --- shared sign: cheaper, but biased (E[g^p] = 0 for odd p) ------------
    sha_m, _ = over_families(
        lambda s: ccs.median(ccs.sharedsign_star(F, 2 * WIDTH,
                                                 np.random.default_rng(s),
                                                 d=DEPTH)))

    print(f"  {'method':<34} {'mean rel. err':>14}")
    print(f"  {'CCS (p=3, complex roots)':<34} {ccs_m:>13.3%}")
    print(f"  {'CCS + T-MCCS truncation':<34} {ccs_t:>13.3%}")
    print(f"  {'per-edge Rademacher (byte-matched)':<34} {per_m:>13.3%}")
    print(f"  {'shared sign (biased, 2x wider)':<34} {sha_m:>13.3%}")
    print("\nAt equal bytes the per-edge real sketch is the strongest entry "
          "here (the\npaper's own reading of Table 4); the biased "
          "constructions are not\ncompetitive at any width -- their median "
          "estimate goes negative in about\nhalf the families.")


def two_join_demo():
    """2-way join: one order-p sketch per relation answers the join."""
    rng = np.random.default_rng(1)
    wts = np.arange(1, U + 1, dtype=np.float64) ** (-ALPHA)
    wts /= wts.sum()
    keys = np.arange(U)
    cR = np.bincount(rng.choice(U, size=300_000, p=wts), minlength=U)
    cS = np.bincount(rng.choice(U, size=300_000, p=wts), minlength=U)
    R = (keys, cR)
    S = (keys, cS)
    truth = float((cR.astype(np.float64) * cS).sum())

    print(f"\n2-way join |R join S|: |U|={U:,} keys, 2 x 300k rows")
    print(f"exact join size = {truth:.6e}\n")
    print(f"  {'method':<34} {'mean rel. err':>14}")

    # Count Sketch: the order-2 specialist (Fast-AGMS inner product).
    cs = np.empty(T)
    for t in range(T):
        est = ccs.cs2_rows(R, S, 2 * WIDTH, np.random.default_rng(3000 + t),
                           d=DEPTH)
        cs[t] = rel(ccs.median(est), truth)
    print(f"  {'Count Sketch (order 2, byte-matched)':<34} {cs.mean():>13.3%}")

    for p in (2, 3, 4, 5):
        e = np.empty(T)
        for t in range(T):
            est = ccs.ccs_cross2_rows(R, S, p, WIDTH,
                                      np.random.default_rng(4000 + t), d=DEPTH)
            e[t] = rel(ccs.median(est), truth)
        print(f"  {f'CCS p={p}':<34} {e.mean():>13.3%}")

    print("\nAll orders are answered from the same per-relation sketches; "
          "the\nexponents +1 and -1 cancel, which is why a single stored "
          "sketch suffices.")


if __name__ == "__main__":
    star_demo()
    two_join_demo()
