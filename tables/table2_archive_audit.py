r"""tables/table2_archive_audit.py -- provenance audit of paper Table 2.

Cross-checks the paper's Table 2 (table:lpnorm) against the ARCHIVED C++
driver outputs bundled in results/table2/: for every (dataset, p) cell, the
relative errors are recomputed from the archived estimates (rounded to 6
significant digits, the drivers' printing precision) against the archived
true values, then aggregated (mean / median over columns) exactly as the
paper reports them.

Result of the audit (all cells "OK" below):
  * result.txt is the p=4 synthetic run, result5.txt the p=5 run; both
    reproduce the paper's synthetic p=4 and p=5 columns exactly.
  * ML{3,4,5}.txt / Airlines{3,4,5}.txt / kasandr{3,4,5}.txt reproduce the
    paper's real-dataset rows exactly (AMS and CCS, mean and median).
  * There is NO archived p=3 synthetic run (result.csv is a stale run on
    older data -- its uniform CCS of 1.43E+13 is a 43% error against a 5.8%
    paper value); the paper's p=3 CCS column is instead reproduced by the
    Python port in tables/table2_lpnorm.py, and its AMS column has no
    archive at all (the C++ AMS samples positions from std::random_device,
    so it is not bit-reproducible in any port).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "table2")

# paper Table 2: (dataset, p) -> (AMS mean, AMS median, CCS mean, CCS median)
PAPER = {
    ("uniform", 4): (2.43e-02, None, 2.55e-01, None),
    ("uniform", 5): (1.94e-02, None, 1.44e-01, None),
    ("zipf1.1", 4): (1.57e-01, None, 6.18e-05, None),
    ("zipf1.1", 5): (9.45e-02, None, 4.43e-04, None),
    ("zipf1.5", 4): (8.37e-03, None, 8.77e-07, None),
    ("zipf1.5", 5): (6.88e-02, None, 1.95e-05, None),
    ("uniform+zipf1.1", 4): (4.60e-03, None, 1.64e-06, None),
    ("uniform+zipf1.1", 5): (3.76e-02, None, 2.17e-05, None),
    ("ML", 3): (1.79e-01, 3.28e-02, 8.06e-03, 5.48e-03),
    ("ML", 4): (1.08e-01, 2.73e-02, 6.36e-03, 1.77e-04),
    ("ML", 5): (2.91e-01, 1.43e-02, 5.80e-03, 5.31e-04),
    ("Airlines", 3): (1.58e-02, 1.70e-02, 7.28e-03, 2.33e-04),
    ("Airlines", 4): (1.53e-02, 1.24e-02, 2.74e-03, 5.60e-05),
    ("Airlines", 5): (3.81e-02, 2.67e-02, 1.49e-03, 1.75e-05),
    ("Kasandr", 3): (3.37e-02, 1.16e-02, 8.71e-03, 3.10e-03),
    ("Kasandr", 4): (6.99e-02, 1.87e-02, 3.41e-02, 4.30e-04),
    ("Kasandr", 5): (4.35e-02, 2.77e-02, 5.08e-02, 1.79e-04),
}


def round6(x):
    return float(f"{x:.5e}")


def read_arch(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) == 4:
            rows.append((parts[0], float(parts[1]), float(parts[2]),
                         float(parts[3])))
    return rows


def read_true(path):
    return [float(x) for x in open(path).read().split()[1:]]


def cell(got, want):
    return "OK " if f"{got:.2E}" == f"{want:.2E}" else "DIFF"


def check(tag, rows, truths, paper):
    ea = [abs(round6(r[1]) - t) / t for r, t in zip(rows, truths)]
    ec = [abs(round6(r[2]) - t) / t for r, t in zip(rows, truths)]
    am, amed = float(np.mean(ea)), float(np.median(ea))
    cm, cmed = float(np.mean(ec)), float(np.median(ec))
    out = [f"{tag:>18}"]
    for nm, got, want in (("AMS mean", am, paper[0]),
                          ("AMS med", amed, paper[1]),
                          ("CCS mean", cm, paper[2]),
                          ("CCS med", cmed, paper[3])):
        if want is not None:
            out.append(f"{nm}={got:.2E} [{cell(got, want)} paper {want:.2E}]")
    print("  ".join(out))


def main():
    print("== synthetic p=4 (result.txt) and p=5 (result5.txt) ==")
    T = {4: read_true(os.path.join(R, "true_value4.csv")),
         5: read_true(os.path.join(R, "true_value5.csv"))}
    names = ["uniform", "zipf1.1", "zipf1.5", "uniform+zipf1.1"]
    for p, arch in ((4, "result.txt"), (5, "result5.txt")):
        rows = read_arch(os.path.join(R, arch))
        for nm, row, tr in zip(names, rows, T[p]):
            ea = abs(round6(row[1]) - tr) / tr
            ec = abs(round6(row[2]) - tr) / tr
            pa, _, pc, _ = PAPER[(nm, p)]
            print(f"  {nm:>16} p={p}: AMS={ea:.2E} [{cell(ea, pa)} paper "
                  f"{pa:.2E}]  CCS={ec:.2E} [{cell(ec, pc)} paper {pc:.2E}]")

    print("\n== synthetic p=3: no archived driver output (result.csv is a "
          "stale run) ==")
    print("   paper p=3 CCS column is reproduced by tables/table2_lpnorm.py; "
          "paper p=3 AMS has no archive (std::random_device positions)")

    print("\n== real datasets (ML / Airlines / Kasandr drivers) ==")
    for ds, stem, truthfile in (("ML", "ML", "ML{p}.csv"),
                                ("Airlines", "Airlines",
                                 "true_Airlines{p}.csv"),
                                ("Kasandr", "kasandr",
                                 "kasandr/true_merge{p}.csv")):
        for p in (3, 4, 5):
            rows = read_arch(os.path.join(R, f"{stem}{p}.txt"))
            truths = read_true(os.path.join(R, truthfile.format(p=p)))
            check(f"{ds} p={p}", rows, truths, PAPER[(ds, p)])


if __name__ == "__main__":
    main()
