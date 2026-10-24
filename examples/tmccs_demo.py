"""T-MCCS: when (and when not) to truncate on the imaginary part.

Run from the artifact root:

    python3 examples/tmccs_demo.py

Every row j of a CCS produces a complex number R_j = sum_s C[j, s]^p whose
real part is the estimate.  The imaginary part is pure collision noise, and
its size predicts how far that row's real part strayed: rows with large |Im|
are the ones that landed far from the truth.  T-MCCS (the paper's Algorithm
alg:tmccs) Tukey-truncates the rows by |Im| -- keep those with
|Im| <= Q3 + 1.5 * IQR, never fewer than half -- and only then takes the
median.

This script shows both sides of that trade, because the gain is real but
bounded:

* on heavily skewed (all-Zipfian) inputs |Im| correlates strongly with the
  error and T-MCCS cuts it substantially;
* on uniform or mildly skewed inputs the correlation is near zero and
  T-MCCS is a tie (the paper's Table 3 reports 0.98-1.03x there).

The last column is the diagnostic to look at before enabling truncation:
if corr(|Im|, error) is small, truncation will not buy anything.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ccs  # noqa: E402

D, N, P, W, DEPTH = 1_000, 100_000, 3, 256, 64
RUNS = 200


def main():
    print(f"F_{P} on truncated Zipf: D={D:,}, N={N:,}, w={W}, d={DEPTH}, "
          f"{RUNS} runs per configuration\n")
    print(f"  {'alpha':>6} {'plain median':>13} {'T-MCCS':>10} {'ratio':>7} "
          f"{'corr(|Im|,err)':>15}")
    for alpha in (0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0):
        freq = ccs.zipf_freq(D, N, alpha, seed=12345)
        truth = float((freq ** P).sum())
        plain = np.empty(RUNS)
        trunc = np.empty(RUNS)
        corr = np.empty(RUNS)
        for r in range(RUNS):
            re, im = ccs.ccs_rows_batch(freq, P, W, DEPTH, seed=900_000 + r)
            plain[r] = abs(ccs.median(re) - truth) / truth
            trunc[r] = abs(ccs.tmccs(re, im) - truth) / truth
            err = np.abs(re - truth)
            if err.std() > 0 and im.std() > 0:
                corr[r] = float(np.corrcoef(im, err)[0, 1])
            else:
                corr[r] = 0.0
        ratio = trunc.mean() / plain.mean()
        flag = "  <- T-MCCS helps" if ratio < 0.9 else ""
        print(f"  {alpha:>6.1f} {plain.mean():>12.3%} {trunc.mean():>9.3%} "
              f"{ratio:>7.2f} {corr.mean():>15.2f}{flag}")
    print("\n(T-MCCS is an enhancement for skewed inputs, not a free lunch: "
          "the\ngain tracks the collision signal in the last column.)")


if __name__ == "__main__":
    main()
