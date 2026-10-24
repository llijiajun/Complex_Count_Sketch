r"""tables/table3_tmccs.py -- reproduce paper Table 3 (tab:tmccs).

Original CCS estimator (plain median) vs T-MCCS and the two controls
Fixed_20 / Rand_20, absolute relative error; alpha is the Zipf skewness;
w=256, d=64, 1000 runs, p=3,4,5.

Faithful port of the archived producer chain
    paper_code/experiments/exp3_tmccs_stats.py        (plain / T-MCCS columns)
    paper_code/python/03_tmccs_estimators.py detail   (Fixed_20 / Rand_20 columns)
with the RNG draw order and seed bases preserved, so the output matches the
archived logs bit-for-bit (see results/exp3_tmccs_stats.expected.txt and
results/tmccs_detail.txt).

Usage:
    python3 tables/table3_tmccs.py            # paper protocol: 1000 runs
    python3 tables/table3_tmccs.py 200        # smoke: 200 runs
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

D, N, W, DEPTH = 1000, 100_000, 256, 64
def _trials_from_argv(default=1000):
    """Paper protocol is 1000 runs; `python3 tables/table3_tmccs.py 200`
    runs a 200-run smoke test.  Import-safe (Jupyter kernels pass flags)."""
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        return int(sys.argv[1])
    return default


TRIALS = _trials_from_argv()
ALPHAS = (0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


def fmt(x):
    """Paper Table 3 number format (2-digit mantissa, E notation)."""
    return f"{x:.2E}"


def main():
    print(f"# Table 3 (tab:tmccs): absolute relative error, w={W}, d={DEPTH}, "
          f"{TRIALS} runs, D={D}, N={N}")
    print(f"# {'alpha':>6} | " + " | ".join(
        f"p={p}: CCS / T-MCCS / Fixed20 / Rand20" for p in (3, 4, 5)))
    summary = {}
    for p in (3, 4, 5):
        print(f"--- p={p} ---", flush=True)
        for alpha in ALPHAS:
            freq = ccs.zipf_freq(D, N, alpha, seed=12345)
            truth = float((freq ** p).sum())
            A = np.empty(TRIALS)
            G = np.empty(TRIALS)
            B = np.empty(TRIALS)
            R = np.empty(TRIALS)
            for t in range(TRIALS):
                re, im = ccs.ccs_rows_batch(freq, p, W, DEPTH, 1000 + t)
                A[t] = abs(ccs.median(re) - truth) / truth
                G[t] = abs(ccs.tmccs(re, im) - truth) / truth
                B[t] = abs(ccs.fixed20(re, im) - truth) / truth
                R[t] = abs(ccs.rand20(re, DEPTH, 1000 + t) - truth) / truth
            wins = int((G < A).sum())
            wlo, whi = wilson(wins, TRIALS)
            summary[(p, alpha)] = (A.mean(), G.mean(), B.mean(), R.mean())
            print(f"a={alpha:.1f}: CCS={fmt(A.mean())} T-MCCS={fmt(G.mean())} "
                  f"Fixed20={fmt(B.mean())} Rand20={fmt(R.mean())}  "
                  f"G/A={G.mean()/A.mean():.3f} wins={wins}/{TRIALS} "
                  f"wilson=[{wlo:.3f},{whi:.3f}]", flush=True)

    # ---- paper-Table-3 facsimile (boldface = best per row and order) -------- #
    print()
    print("# paper Table 3 facsimile (bold = best of the four per row/order)")
    header = "alpha | " + " | ".join(
        f"p={p}: CCS T-MCCS Fixed20 Rand20" for p in (3, 4, 5))
    print(f"# {header}")
    for alpha in ALPHAS:
        cells = []
        for p in (3, 4, 5):
            vals = summary[(p, alpha)]
            best = min(vals)
            cells.append(" ".join(
                (f"**{fmt(v)}**" if v == best else fmt(v)) for v in vals))
        print(f"{alpha:.1f} | " + " | ".join(cells))

    # ---- Section 5.3 text numbers ------------------------------------------ #
    print()
    print("# Section 5.3 headline numbers")
    for p in (3, 4, 5):
        red = []
        for alpha in (1.1, 1.5, 2.0):
            A, G = summary[(p, alpha)][0], summary[(p, alpha)][1]
            red.append(1 - G / A)
        print(f"p={p}: T-MCCS cut over alpha in [1.1,2.0]: "
              + " ".join(f"{100*r:.0f}%" for r in red))
    for alpha in (0.0, 0.5):
        worst = max(summary[(p, alpha)][1] / summary[(p, alpha)][0]
                    for p in (3, 4, 5))
        print(f"alpha={alpha}: worst T-MCCS/plain over p=3,4,5 = {worst:.3f}")


if __name__ == "__main__":
    main()
