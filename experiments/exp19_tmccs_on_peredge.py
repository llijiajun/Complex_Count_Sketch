r"""Exp 19 -- does T-MCCS's truncation port to the real per-edge sketch?

The per-edge sketch has no imaginary part, so the |Im(X_j)| signal is not
available.  The closest real analogue is a per-row ENERGY statistic computed
from the SAME stored counters,
E_j = sum_s (C_1 C_2 C_3)[s]^2,
which flags buckets holding large product mass.  We test whether Tukey
truncation on E_j (drop the 20% highest-energy rows) improves the per-edge
median: it could flag collision-heavy rows, or it could bias the estimate,
because E_j also contains the true join mass.  Per family we also report the
Spearman correlation of E_j with |R_j - truth| (per-row signal quality).

Reference on the same data: CCS plain and CCS + T-MCCS.

Protocol: zipf1.1 x3, p=3, w=256, d=64, 30 families.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column, merge_universe
import ccs

P, W, DEPTH, T = 3, 256, 64, 30

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

if __name__ == "__main__":
    fc = [lp_column("zipf1.1.dat" + s) for s in ("", "2", "3")]
    allk, F = merge_universe(fc)
    truth = float((F[0] * F[1] * F[2]).sum())
    U = len(allk)

    plain, trunc, bias_t = [], [], []
    rhos = []
    for t in range(T):
        rng = np.random.default_rng(8100 + t)
        rows = np.empty(DEPTH)
        energies = np.empty(DEPTH)
        kept_means = []
        for j in range(DEPTH):
            h = rng.integers(0, 2 * W, size=U)
            s12 = rng.integers(0, 2, size=U) * 2 - 1
            s13 = rng.integers(0, 2, size=U) * 2 - 1
            signs = [s12 * s13, s12, s13]                # col1 = product of edges
            cs = [np.bincount(h, weights=f * sg, minlength=2 * W)
                  for f, sg in zip(F, signs)]
            prod = cs[0] * cs[1] * cs[2]
            rows[j] = float(prod.sum())
            energies[j] = float((prod ** 2).sum())
        plain.append(abs(ccs.median(rows) - truth) / truth)
        # Tukey truncation on the energy signal
        q1, q3 = np.percentile(energies, [25, 75])
        keep = np.nonzero(energies <= q3 + 1.5 * (q3 - q1))[0]
        if len(keep) < DEPTH // 2:
            keep = np.argsort(energies)[: DEPTH // 2]
        trunc.append(abs(float(np.median(rows[keep])) - truth) / truth)
        bias_t.append(abs(rows[keep].mean() - truth) / truth)
        rhos.append(spearman(energies, np.abs(rows - truth) / truth))

    # reference: CCS plain and CCS + T-MCCS on the same data
    A, G = [], []
    for t in range(T):
        rng = np.random.default_rng(7000 + t)
        re, im = ccs.star_rows(F, P, W, rng, d=DEPTH)
        A.append(abs(ccs.median(re) - truth) / truth)
        G.append(abs(ccs.tmccs(re, im) - truth) / truth)

    print(f"per-edge plain median:                mean rel err = {np.mean(plain):.4g}")
    print(f"per-edge energy-truncated (top-20%):  mean rel err = {np.mean(trunc):.4g}")
    print(f"truncated mean-of-rows |bias|/truth:  {np.mean(bias_t):.4g}")
    print(f"per-family Spearman(E_j, |row err|):  mean = {np.mean(rhos):.3f}")
    print(f"reference CCS: plain = {np.mean(A):.4g}   T-MCCS = {np.mean(G):.4g}")
    print("EXP19_DONE")
