r"""experiments/exp3b_aggregators.py -- row-aggregator comparison for CCS.

Original: paper_code/experiments/exp3b_aggregators.py
(paper Section 5.3, "Against other aggregators").

On the Table 3 protocol (D=1000, N=1e5, w=256, d=64, 1000 trials, p=3,4,5),
four aggregators of the SAME row estimates X_j = Re(sum_s C[j,s]^p):

  median          plain median over the d rows (the original CCS estimator)
  mean            arithmetic mean over the d rows
  med-of-means    d rows split into 8 groups of 8; median of the 8 group means
  T-MCCS          Tukey truncation on |Im(X_j)| (>= half the rows kept), median

Motivation: the unbiasedness of the row estimator does not transfer to every
aggregator; the median is in general biased (it converges to the median of
the row distribution). This experiment quantifies all three biases on the
same sketches and shows where the median family beats the mean (skewed data).

Artifact changes vs the original: the 03_tmccs_estimators module is replaced
by ccs.py (zipf_freq / ccs_rows_batch / tmccs / med_of_means -- the same
protocols, verified bit-exact against the archived logs).  Output matches
results/experiments/exp3b_aggregators.txt.

Usage:
    python3 experiments/exp3b_aggregators.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

D, N, w, d = 1000, 100_000, 256, 64


def main():
    trials = 1000
    for p in (3, 4, 5):
        print(f"--- p={p} ---", flush=True)
        for a in (0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0):
            freq = ccs.zipf_freq(D, N, a, seed=12345)
            truth = float(np.sum(freq ** p))
            A = np.empty(trials); B = np.empty(trials)
            C = np.empty(trials); G = np.empty(trials)
            for t in range(trials):
                re, im_row = ccs.ccs_rows_batch(freq, p, w, d, 1000 + t)
                A[t] = abs(ccs.median(re) - truth) / truth
                B[t] = abs(ccs.mean(re) - truth) / truth
                C[t] = abs(ccs.med_of_means(re) - truth) / truth
                G[t] = abs(ccs.tmccs(re, im_row) - truth) / truth
            print(f"a={a:.1f}: median={A.mean():.4e} mean={B.mean():.4e} "
                  f"medofmeans={C.mean():.4e} tmccs={G.mean():.4e} "
                  f"mean/med={B.mean()/A.mean():.2f} MoM/med={C.mean()/A.mean():.2f} "
                  f"T/med={G.mean()/A.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
    print("EXP3B_DONE", flush=True)
