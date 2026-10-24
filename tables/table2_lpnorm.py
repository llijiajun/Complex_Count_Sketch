r"""tables/table2_lpnorm.py -- reproduce paper Table 2 (table:lpnorm).

F_p estimation error of AMS vs CCS on the four synthetic Lp columns and the
three real datasets, p = 3, 4, 5.

Provenance (see README "Table 2" for the full audit)
----------------------------------------------------
The paper's numbers come from the C++ drivers in cpp/ (main.cpp, ML.cpp,
Airlines.cpp, kasandr.cpp), each run for 5 rounds with hash_seed = round and
a running mean of the per-round estimates.  This script re-runs the SAME
protocol through the bit-exact Python port of those drivers (ccs.py:

  * cpp_ccs_estimate  -- ComplexCountSketch.estimate_lp(): width
    ceil(2^p/eps^2), depth ceil(ln(4/gam)), MurmurHash32 bucket/root hashes,
    C^p by repeated complex multiplication, median over rows.
  * cpp_ams_estimate  -- AMS positional sample-count estimator.

Reproduction status, verified cell by cell:
  * CCS column: reproduced EXACTLY for every cell of the paper's table.
  * AMS column: the C++ draws its sampled positions from std::random_device
    (not reproducible); the archived driver outputs (results/) are the
    paper's actual source and are cross-referenced below, and the Python
    port (deterministic position sampler) is statistically equivalent.
  * Relative errors are computed from the estimates rounded to 6 significant
    digits, exactly as the C++ drivers print them and as the paper's table
    was assembled (this matters for the 1e-6-level cells: the rounding is
    worth up to 5e-7 relative).

Real datasets: mean and median of the per-column relative errors, as in the
paper.  Column orders follow the C++ drivers.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
from loaders import (lp_lines, ml_ratings_lines, airlines_lines,
                     kasandr_column)

ROUNDS = 5           # C++ drivers: for (i = 0; i < round; i++) with round = 5
GAM = 0.01           # delta = 0.01 -> depth = ceil(ln(400)) = 6
REAL = ("MovieLens", "Airlines", "Kasandr")

# name -> (columns, eps, N, loader)
DATASETS = {
    "uniform": (["uniform.dat"], 0.1, 10_000_000, lp_lines),
    "zipf1.1": (["zipf1.1.dat"], 0.1, 10_000_000, lp_lines),
    "zipf1.5": (["zipf1.5.dat"], 0.1, 10_000_000, lp_lines),
    "uniform+zipf1.1": (["uniform+zipf1.1.dat"], 0.1, 10_000_000, lp_lines),
    "MovieLens": (["userId", "movieId", "rating"], 0.05, 25_000_096,
                  ml_ratings_lines),
    "Airlines": (["DepDelay", "Month", "DayofMonth", "DayOfWeek",
                  "CRSDepTime", "CRSArrTime", "UniqueCarrier", "Origin",
                  "Dest", "Distance"], 0.05, 10_000_000, airlines_lines),
    "Kasandr": (["rating", "utcdate", "merchant", "category", "countrycode",
                 "offerid", "userid"], 0.05, 17_764_278, kasandr_column),
}

# Paper Table 2 (table:lpnorm).  Synthetic rows: (AMS, CCS) per p.  Real
# datasets: ((AMS mean, AMS median), (CCS mean, CCS median)) per p.
PAPER = {
    "uniform": {3: (8.34e-03, 5.81e-02), 4: (2.43e-02, 2.55e-01),
                5: (1.94e-02, 1.44e-01)},
    "zipf1.1": {3: (3.55e-02, 1.52e-04), 4: (1.57e-01, 6.18e-05),
                5: (9.45e-02, 4.43e-04)},
    "zipf1.5": {3: (5.41e-03, 4.22e-06), 4: (8.37e-03, 8.77e-07),
                5: (6.88e-02, 1.95e-05)},
    "uniform+zipf1.1": {3: (3.14e-02, 3.34e-06), 4: (4.60e-03, 1.64e-06),
                        5: (3.76e-02, 2.17e-05)},
    "MovieLens": {3: ((1.79e-01, 3.28e-02), (8.06e-03, 5.48e-03)),
                  4: ((1.08e-01, 2.73e-02), (6.36e-03, 1.77e-04)),
                  5: ((2.91e-01, 1.43e-02), (5.80e-03, 5.31e-04))},
    "Airlines": {3: ((1.58e-02, 1.70e-02), (7.28e-03, 2.33e-04)),
                 4: ((1.53e-02, 1.24e-02), (2.74e-03, 5.60e-05)),
                 5: ((3.81e-02, 2.67e-02), (1.49e-03, 1.75e-05))},
    "Kasandr": {3: ((3.37e-02, 1.16e-02), (8.71e-03, 3.10e-03)),
                4: ((6.99e-02, 1.87e-02), (3.41e-02, 4.30e-04)),
                5: ((4.35e-02, 2.77e-02), (5.08e-02, 1.79E-04))},
}


def round6(x):
    """6 significant digits, exactly as the C++ drivers print estimates."""
    return float(f"{x:.5e}")


def run_dataset(name, ps=(3, 4, 5)):
    """Per-column (truth, AMS estimate, CCS estimate) under the C++ protocol."""
    cols, eps, N_cpp, loader = DATASETS[name]
    out = {}
    for col in cols:
        if loader is kasandr_column:
            keys, counts, N = loader(col)
        else:
            keys, counts = loader(col)
            N = int(counts.sum())
        if N != N_cpp:
            print(f"  [note] {name}/{col}: {N} rows in the bundled data vs "
                  f"N={N_cpp} in the C++ driver (AMS samples positions in "
                  f"[0,N); the CCS column does not depend on N)", flush=True)
        keys = list(keys)
        depth = ccs.cpp_dims(3, eps, GAM)[1]
        # hashes are p-independent: compute once per round, reuse across p
        per_round = [ccs.cpp_ccs_hashes(keys, i, depth) for i in range(ROUNDS)]
        for p in ps:
            truth = float((counts.astype(np.float64) ** p).sum())
            ams_acc = ccs_acc = 0.0
            for i in range(ROUNDS):
                a = ccs.cpp_ams_estimate(keys, counts, N, p, eps, GAM, i,
                                         seed=i)
                ams_acc = (ams_acc * i + a) / (i + 1)
                c = ccs.cpp_ccs_estimate(keys, counts, p, eps, GAM, i,
                                         hashes=per_round[i])
                ccs_acc = (ccs_acc * i + c) / (i + 1)
            out[(col, p)] = dict(
                truth=truth, ams=ams_acc, ccs=ccs_acc,
                ea=abs(round6(ams_acc) - truth) / truth,
                ec=abs(round6(ccs_acc) - truth) / truth)
        print(f"  [{name}] {col}: D={len(keys)}  " + "  ".join(
            f"p={p}: ams={out[(col, p)]['ea']:.2E} ccs={out[(col, p)]['ec']:.2E}"
            for p in ps), flush=True)
    return out


def aggregate(name, res):
    """Per p: (AMS mean, AMS median, CCS mean, CCS median) as the paper reports."""
    cols = DATASETS[name][0]
    return {p: (float(np.mean([res[(c, p)]["ea"] for c in cols])),
                float(np.median([res[(c, p)]["ea"] for c in cols])),
                float(np.mean([res[(c, p)]["ec"] for c in cols])),
                float(np.median([res[(c, p)]["ec"] for c in cols])))
            for p in (3, 4, 5)}


def cell_ok(got, want):
    """The paper's cells are 2-digit E-notation: match means the printed
    strings agree (all values are compared at the paper's precision)."""
    return f"{got:.2E}" == f"{want:.2E}"


def fmt_pair(vals):
    return ",".join(f"{v:.2E}" for v in vals)


def main(which=None):
    """Run the Table 2 protocol.

    `which` selects datasets (default: all).  Import-safe: a Jupyter kernel's
    sys.argv is ignored -- pass the dataset list explicitly instead.
    """
    if which is None:
        which = list(DATASETS)
    t0 = time.time()
    print(f"# Table 2 (table:lpnorm): F_p relative error, AMS vs CCS, "
          f"{ROUNDS} rounds, running mean", flush=True)
    print("# relative errors use the 6-significant-digit rounded estimates "
          "(the C++ printing convention the paper's table was built from)",
          flush=True)
    aggs = {}
    for name in which:
        print(f"== {name} ==", flush=True)
        aggs[name] = aggregate(name, run_dataset(name))

    # ---- paper facsimile: reproduced (Python port) vs published ------------- #
    print("\n# paper Table 2 facsimile: reproduced / published "
          "(* = match within rounding, ! = differs)")
    print(f"# {'dataset':>16} | " + " | ".join(
        f"p={p} AMS / CCS" for p in (3, 4, 5)))
    for name in which:
        agg, paper = aggs[name], PAPER[name]
        cells = []
        for p in (3, 4, 5):
            am, amed, cm, cmed = agg[p]
            if name in REAL:
                pa, pc = paper[p]
                got_a, got_c = (am, amed), (cm, cmed)
            else:
                pa, pc = (paper[p][0],), (paper[p][1],)
                got_a, got_c = (am,), (cm,)
            a_ok = all(cell_ok(g, w) for g, w in zip(got_a, pa))
            c_ok = all(cell_ok(g, w) for g, w in zip(got_c, pc))
            cells.append(f"{fmt_pair(got_a)}{'*' if a_ok else '!'} / "
                         f"{fmt_pair(pa)}    {fmt_pair(got_c)}"
                         f"{'*' if c_ok else '!'} / {fmt_pair(pc)}")
        print(f"  {name:>16} | " + " | ".join(cells), flush=True)

    # ---- boldface facsimile (paper layout, best per row/order) -------------- #
    print("\n# paper Table 2 facsimile (reproduced values; "
          "bold = best of AMS/CCS per row/order)")
    print(f"# {'dataset':>16} {'agg':>7} | " + " | ".join(
        f"p={p} AMS CCS" for p in (3, 4, 5)))
    for name in which:
        agg = aggs[name]
        rows = [("", None)] if name not in REAL else [("Mean", 0),
                                                      ("Median", 1)]
        for label, idx in rows:
            cells = []
            for p in (3, 4, 5):
                am, amed, cm, cmed = agg[p]
                a = am if idx is None else (am, amed)[idx]
                c = cm if idx is None else (cm, cmed)[idx]
                best = min(a, c)
                cells.append(" ".join(
                    (f"**{v:.2E}**" if v == best else f"{v:.2E}")
                    for v in (a, c)))
            print(f"  {name:>16} {label:>7} | " + " | ".join(cells))

    print(f"\n# total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
