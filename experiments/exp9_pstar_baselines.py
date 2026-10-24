r"""experiments/exp9_pstar_baselines.py -- real-valued baselines for the
p-star join (paper Section 5.4, Table 4's baseline columns).

Original: paper_code/experiments/exp9_pstar_baselines.py

Runs, on exactly the Table 4 protocol (p=3, w=256, d=64; synthetic 100 hash
families, real 24), three real-valued competitors:

  peredge   one Rademacher sign per EDGE of the 3-node star (edges (1,2),(1,3));
            column t carries the product of its incident edge signs, so the
            diagonal contributes E[s12^2 s13^2] = 1: UNBIASED. This is the
            order-2 per-edge construction of Proposition peredge, run at
            byte-matched width 2w = 512 real counters (CCS: 256 complex).
  sharedsign one Rademacher sign shared by all three columns (the direct real
            analog of CCS): E[R] = sum_a f1 f2 f3 E[g^3] = 0, biased.
  cubes     sum of CUBES of a real Count Sketch counter array built on the
            elementwise-summed stream (the brute-force estimator the paper's
            setup paragraph mentions): E[g^3] = 0, biased.

For each dataset we report, averaged over hash families: mean/median relative
error of the plain median over rows, the fraction of families whose median
estimate is negative, and (for the unbiased per-edge variant) the relative
deviation of the average estimate from the truth.

Artifact changes vs the original: data loading goes through loaders.py
(bundled gzip columns, byte-identical to the originals) and the estimators
through ccs.py (star_rows / peredge_star / sharedsign_star / cubes_real --
same protocols and seeds, verified bit-exact against the archived logs;
the synthetic combos also appear in tables/table4_pstar.py).  Output matches
results/experiments/exp9_{synth2..5,ml,kas}.txt numerically; the archived
synth tags were captured with each dataset name truncated to 6 characters
('uniforxzipf1.xzipf1.', and zipf1.1/zipf1.5 both truncate to 'zipf1.'),
while this script prints full names -- so run_all.py's check_exp9 keys the
comparison on the true join size instead of the tag.

Usage:
    python3 experiments/exp9_pstar_baselines.py           # all
    python3 experiments/exp9_pstar_baselines.py synth     # 6 synthetic combos
    python3 experiments/exp9_pstar_baselines.py synth2    # single combo
    python3 experiments/exp9_pstar_baselines.py ml        # MovieLens 3-star
    python3 experiments/exp9_pstar_baselines.py kas       # Kasandr 3-star
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
import loaders

P = 3
W, D_ROWS = 256, 64


def report(tag, truth, F, T=100):
    Fsum = F[0] + F[1] + F[2]
    out = [f"[{tag}] truth={truth:.6g}"]
    # CCS reference (plain median), same protocol as Table 4
    A = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(7000 + t)
        re, _ = ccs.star_rows(F, P, W, rng, d=D_ROWS)
        A[t] = abs(ccs.median(re) - truth) / truth
    out.append(f"CCS mean={A.mean():.4g}")
    # per-edge real (unbiased), byte-matched width
    E = np.empty(T); NEG = 0; dev = 0.0
    for t in range(T):
        rng = np.random.default_rng(8100 + t)
        est = ccs.peredge_star(F, W, rng, d=D_ROWS)
        m = float(np.median(est))
        NEG += (m < 0)
        dev += abs(est.mean() - truth) / truth
        E[t] = abs(m - truth) / truth
    out.append(f"peredge mean={E.mean():.4g} med={np.median(E):.4g} neg={NEG}/{T} bias={dev/T:.3g}")
    # shared-sign real product (biased)
    S = np.empty(T); NEG2 = 0
    for t in range(T):
        rng = np.random.default_rng(8200 + t)
        est = ccs.sharedsign_star(F, W, rng, d=D_ROWS)
        m = float(np.median(est))
        NEG2 += (m < 0)
        S[t] = abs(m - truth) / truth
    out.append(f"sharedsign mean={S.mean():.4g} med={np.median(S):.4g} neg={NEG2}/{T}")
    # cubes of one real sketch (biased)
    C = np.empty(T); NEG3 = 0
    for t in range(T):
        rng = np.random.default_rng(8300 + t)
        est = ccs.cubes_real(Fsum, W, rng, d=D_ROWS)
        m = float(np.median(est))
        NEG3 += (m < 0)
        C[t] = abs(m - truth) / truth
    out.append(f"cubes mean={C.mean():.4g} med={np.median(C):.4g} neg={NEG3}/{T}")
    print("  ".join(out), flush=True)


COMBOS = [
    ("uniform", "uniform", "uniform"),
    ("uniform", "uniform", "zipf1.1"),
    ("uniform", "zipf1.1", "zipf1.5"),
    ("uniform", "zipf1.5", "zipf1.5"),
    ("zipf1.1", "zipf1.1", "zipf1.1"),
    ("zipf1.5", "zipf1.5", "zipf1.5"),
]
SUFS = ["", "2", "3"]


def synth():
    for names in COMBOS:
        fc = [loaders.lp_column(t + ".dat" + s) for t, s in zip(names, SUFS)]
        allk, F = loaders.merge_universe(fc)
        truth = loaders.star_truth(fc)
        report("synth " + "x".join(names), truth, F)


def reindex(k, c, kcol):
    f = np.zeros(len(kcol), dtype=np.float64)
    f[np.searchsorted(kcol, k)] = c
    return f


def real_ml():
    for key in ("userId", "movieId"):
        arr = loaders.raw(os.path.join("ML", "ratings", key))
        kcol = np.unique(arr)
        shards = [np.unique(arr[i::3], return_counts=True) for i in range(3)]
        F = [reindex(k, c, kcol) for k, c in shards]
        truth = float((F[0] * F[1] * F[2]).sum())
        report(f"3star ML {key}", truth, F, T=24)


def real_kasandr():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import exp2_kasandr_3star as K
    users, cats, times, U = K.load()
    # aligned: three time-window shards
    qs = np.percentile(times, [100 / 3, 200 / 3])
    m1, m2 = times <= qs[0], (times > qs[0]) & (times <= qs[1])
    m3 = times > qs[1]
    k, c = np.unique(users[m1], return_counts=True)
    F1 = np.zeros(U); F1[k] = c
    k, c = np.unique(users[m2], return_counts=True)
    F2 = np.zeros(U); F2[k] = c
    k, c = np.unique(users[m3], return_counts=True)
    F3 = np.zeros(U); F3[k] = c
    report("3star KAS time", float((F1 * F2 * F3).sum()), [F1, F2, F3], T=24)
    # misaligned: top-3 categories
    ck, cc = np.unique(cats, return_counts=True)
    top = ck[np.argsort(-cc)[:3]]
    Fs = []
    for cv in top:
        k, c = np.unique(users[cats == cv], return_counts=True)
        f = np.zeros(U); f[k] = c
        Fs.append(f)
    report("3star KAS cat", float((Fs[0] * Fs[1] * Fs[2]).sum()), Fs, T=24)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        synth(); real_ml(); real_kasandr()
    elif which == "synth":
        synth()
    elif which.startswith("synth"):  # synth2..synth5 : single combo by index
        idx = int(which[5:])
        names = COMBOS[idx]
        fc = [loaders.lp_column(t + ".dat" + s) for t, s in zip(names, SUFS)]
        allk, F = loaders.merge_universe(fc)
        truth = loaders.star_truth(fc)
        report("synth " + "x".join(names), truth, F)
    elif which == "ml":
        real_ml()
    elif which == "kas":
        real_kasandr()
    print("EXP9_DONE", flush=True)
