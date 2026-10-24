r"""tables/table5_realccs.py -- reproduce paper Table 5 (tab:realccs).

Can one stored order-p CCS sketch also answer 2-join queries?  Mean relative
error on four real 2-joins (exact sizes by full scans): Count Sketch (the
order-2 specialist, w=2048 real counters) and JoinSketch (same counter
budget, corrected key interface) against CCS sketches built at orders 3, 4,
5 (w=1024 complex counters, byte-matched, d=64, 24 families).  Plus the
aligned real 3-star control (Section 5.5).

Provenance
----------
Faithful port of paper_code/experiments/exp1_real_joins.py, whose archived
output (results/exp1_real_joins.out) is the paper's source for the CS column
and for the 3-star rows:

  * 3-star: seeds 9000+t, T=24, p=3, w=256, d=64; plain median + T-MCCS.
  * CS 2-join baseline: seeds 3000+t, w=2048, d=64, T=24.
  * CCS 2-join: seeds 4000+p+t, w=1024, d=64, T=24.

KNOWN PROVENANCE GAP (honest disclosure, also in the README): the paper's
CCS p=3/4/5 column does NOT match this archived protocol -- the published
run was never archived (both the 4000+p protocol here and the later
4000+100p protocol of the release tree disagree with the paper, e.g. AL
Origin-Dest p=3: paper 2.5E-03 vs archived 2.97E-03).  The JoinSketch column
is an external C++ baseline (wang2023joinsketch, corrected key interface)
not implemented in this artifact; its published values are quoted as
reference data only.

Usage:
    python3 tables/table5_realccs.py            # full protocol (T=24)
    python3 tables/table5_realccs.py 8          # smoke (numbers drift)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
from loaders import ml_ratings, ml_tags, airlines, raw

def _families_from_argv(default=24):
    """Paper protocol is T=24; `python3 tables/table5_realccs.py 8` runs a
    smoke test.  Import-safe (Jupyter kernels pass flags)."""
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        return int(sys.argv[1])
    return default


T = _families_from_argv()
W_STAR, W_CS, W_CCS2, D = 256, 2048, 1024, 64
P = 3

# Paper Table 5 (tab:realccs): join -> (size, CS, JoinSketch, CCS p3, p4, p5)
PAPER = {
    "ML userId":    (1.24e+09, 4.5e-02, 1.8e-02, 4.5e-02, 4.7e-02, 4.6e-02),
    "ML movieId":   (1.09e+10, 5.4e-03, 3.9e-03, 4.4e-03, 4.3e-03, 4.4e-03),
    "AL Origin-Dest": (1.92e+12, 6.3e-04, 7.5e-04, 2.5e-03, 1.2e-03, 1.5e-03),
    "AL Arr-Dep":   (1.78e+11, 4.0e-03, 5.1e-03, 5.0e-03, 4.1e-03, 5.4e-03),
}
# exp1's hardcoded join sizes (verified against the bundled data below)
EXP1_TRUTH = {
    "ML userId": 1243199075, "ML movieId": 10883253790,
    "AL Origin-Dest": 1922584284057, "AL Arr-Dep": 177521585278,
}


def reindex(k, c, kcol):
    f = np.zeros(len(kcol), dtype=np.int64)
    f[np.searchsorted(kcol, k)] = c
    return f


def three_star(key, arr, tag):
    """Aligned real 3-star: round-robin shards of one column (exp1 part A)."""
    kcol = np.unique(arr)
    shards = [np.unique(arr[i::3], return_counts=True) for i in range(3)]
    F = [reindex(k, c, kcol) for k, c in shards]
    truth = float((F[0].astype(object) * F[1] * F[2]).sum())
    A = np.empty(T); G = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(9000 + t)
        re, im = ccs.star_rows([f.astype(np.float64) for f in F], P, W_STAR,
                               rng, d=D)
        A[t] = abs(ccs.median(re) - truth) / truth
        G[t] = abs(ccs.tmccs(re, im) - truth) / truth
    print(f"[3star ML {key}] truth={truth:.6g} plain mean={A.mean():.4f} "
          f"med={np.median(A):.4f} T-MCCS mean={G.mean():.4f} "
          f"med={np.median(G):.4f} win={float(np.mean(G < A)):.2f}", flush=True)
    return truth


def join_size(R, S):
    """Exact 2-join size: sum over common keys of the count product."""
    allk = np.union1d(R[0], S[0])
    posR = np.searchsorted(allk, R[0])
    posS = np.searchsorted(allk, S[0])
    fR = np.zeros(len(allk), dtype=np.int64); fR[posR] = R[1]
    fS = np.zeros(len(allk), dtype=np.int64); fS[posS] = S[1]
    return int((fR.astype(object) * fS).sum())


def two_join(R, S, name):
    truth = join_size(R, S)
    ok = "OK" if truth == EXP1_TRUTH[name] else \
        f"DIFF (exp1 hardcoded {EXP1_TRUTH[name]})"
    print(f"[2join {name}] truth={truth:.6g} ({ok})", flush=True)
    cs = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(3000 + t)
        est = ccs.cs2_rows(R, S, W_CS, rng, d=D)
        cs[t] = abs(ccs.median(est) - truth) / truth
    line = [f"CS(w={W_CS}) mean={cs.mean():.3e} med={float(np.median(cs)):.3e}"]
    ccs_means = {}
    for p in (3, 4, 5):
        E = np.empty(T)
        for t in range(T):
            rng = np.random.default_rng(4000 + p + t)
            est = ccs.ccs_cross2_rows(R, S, p, W_CCS2, rng, d=D)
            E[t] = abs(ccs.median(est) - truth) / truth
        ccs_means[p] = E.mean()
        line.append(f"CCS p={p} mean={E.mean():.3e} "
                    f"med={float(np.median(E)):.3e}")
    print("  ".join(line), flush=True)
    return truth, cs.mean(), ccs_means


def main():
    t0 = time.time()
    print(f"# Table 5 (tab:realccs): T={T} families, d={D}, "
          f"CS w={W_CS}, CCS w={W_CCS2}, 3-star w={W_STAR}", flush=True)

    # ---------- (A) aligned real 3-star: round-robin shards of ratings ------ #
    print("loading MovieLens ratings columns...", flush=True)
    ru = raw("ML/ratings/userId.gz")
    rm = raw("ML/ratings/movieId.gz")
    for key, arr in (("userId", ru), ("movieId", rm)):
        three_star(key, arr, None)
    del ru, rm

    # ---------- (B) real 2-joins: CS baseline + CCS at orders p=3..5 -------- #
    print("loading join columns...", flush=True)
    R_u = ml_ratings("userId"); T_u = ml_tags("userId")
    R_m = ml_ratings("movieId"); T_m = ml_tags("movieId")
    O = airlines("Origin"); Dst = airlines("Dest")
    A = airlines("CRSArrTime"); Dp = airlines("CRSDepTime")
    joins = [(R_u, T_u, "ML userId"), (R_m, T_m, "ML movieId"),
             (O, Dst, "AL Origin-Dest"), (A, Dp, "AL Arr-Dep")]
    results = {}
    for R, S, name in joins:
        results[name] = two_join(R, S, name)

    # ---- paper Table 5 facsimile: reproduced vs published ------------------- #
    print("\n# paper Table 5 facsimile: reproduced / published "
          "(* = match at 2 digits, ! = differs)")
    print(f"# {'join':>16} | {'size':>9} {'paper':>9} | {'CS':>9} {'paper':>9} "
          f"| {'JS':>9} {'paper':>9} | " + " | ".join(
              f"CCS p={p}" for p in (3, 4, 5)))
    # The paper's Table 5 prints 2 significant digits (4.5E-02), unlike
    # Tables 2/3 which print 2-digit mantissas (5.81E-02).
    for name, (truth, cs_mean, ccs_means) in results.items():
        psz, pcs, pjs, pp3, pp4, pp5 = PAPER[name]
        star = lambda got, want: "*" if f"{got:.1E}" == f"{want:.1E}" else "!"
        print(f"  {name:>16} | {truth:.2E}{star(truth, psz)} {psz:.1E} | "
              f"{cs_mean:.2E}{star(cs_mean, pcs)} {pcs:.1E} | "
              f"{'--':>9} {pjs:.1E} | "
              + " | ".join(f"{ccs_means[p]:.2E}{star(ccs_means[p], pw)} "
                           f"{pw:.1E}"
                           for p, pw in zip((3, 4, 5), (pp3, pp4, pp5))),
              flush=True)
    print("\n# JoinSketch column is an external C++ baseline (corrected key "
          "interface); published values quoted as reference, not reproduced "
          "here.  The paper's CCS column came from an unarchived run -- see "
          "the module docstring's provenance gap note.")
    print(f"\n# total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
