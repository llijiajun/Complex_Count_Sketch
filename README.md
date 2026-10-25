# CCS artifact — Complex Count Sketch

Reproduction artifact for **"Complex Count Sketch: Mergeable
Frequency-Moment and p-Star Join Size Estimation"** (PVLDB submission).

Everything the paper's tables and experiments need, self-contained: the
scripts, the data, and the archived production logs that produced the
published numbers.  The scattered scripts of the original repository
(`paper_code/python/`, `paper_code/experiments/`, `paper_code/cpp/`,
`experiment/`) are reorganised here into five layers.

```
Complex_Count_Sketch/
├── ccs.py               # core library: hashing / sketches / estimators /
│                        #   T-MCCS / exact verification utilities /
│                        #   user-facing API (ComplexCountSketch)
├── loaders.py           # data layer: uniform accessors for the bundled
│                        #   gzip column files
├── examples/            # standalone, runnable CCS demos (start here!)
├── tests/               # test_api.py: pins the user-facing API to the
│                        #   paper's own estimator bit-for-bit
├── tables/              # paper Tables 2-5, Figures 1-2, cell-by-cell audit
│   ├── table2_lpnorm.py         Table 2 (4 synthetic + ML/Airlines/Kasandr)
│   ├── table2_archive_audit.py  Table 2: paper values vs C++ archived logs
│   ├── table3_tmccs.py          Table 3 (CCS vs T-MCCS vs Fixed20/Rand20)
│   ├── table4_pstar.py          Table 4 (p-star: CCS/T-MCCS/G-A + baselines)
│   ├── table5_realccs.py        Table 5 (real 3-star + 2-join, orders p>=3)
│   ├── figures.py               Figure 1 (width sweep) + Figure 2 (regime)
│   └── figs_out/                rendered figures (PDF + PNG)
├── experiments/         # the paper's other 16 experiments (exp2..exp18)
├── cpp/                 # original 2024 C++ pipeline (Table 2, Table 5 ref.)
├── data/                # all data (409 MB, gzip column views, byte-identical)
└── results/             # archived production logs + this port's outputs
    ├── table{2,3,4,5}/  archived logs for the paper tables (audit baseline)
    ├── experiments/     archived logs for the paper experiments
    └── figures/         archived data for Figure 1 (exp8b)
```

## Quick start

Environment: Python 3.11 + numpy 1.26.4 + matplotlib 3.10 (see
"Environment").  No network access is needed; all data is bundled.

```bash
cd Complex_Count_Sketch

# --- use CCS on your own data -------------------------------------------
python3 examples/quickstart.py       # ~3 s    F_p of a stream, in 3 lines
python3 examples/join_demo.py        # ~30 s   3-way star and 2-way joins
python3 examples/tmccs_demo.py       # ~2 s    when truncating on |Im| pays
python3 examples/one_deployment.py   # ~2 s    one sketch, three tasks
python3 tests/test_api.py            # ~20 s   API consistency checks

# --- reproduce the paper's tables (each script runs standalone) ---------
python3 tables/table2_lpnorm.py            # Table 2, ~8 min
python3 tables/table3_tmccs.py             # Table 3, ~4 min
python3 tables/table4_pstar.py             # Table 4, ~45 min
python3 tables/table5_realccs.py           # Table 5, ~6 min
python3 tables/figures.py                  # Figures 1/2 from archives, seconds
python3 tables/figures.py --recompute      # recompute + verify, ~2 min

# --- the paper's other experiments --------------------------------------
python3 experiments/run_all.py --list      # the 16 steps and their cost
python3 experiments/run_all.py exp11 exp16 # run chosen steps, then audit
python3 experiments/run_all.py             # everything (~2.5 h)

# --- C++ (needs g++) ----------------------------------------------------
bash experiments/exp5_join2_matched.sh     # JoinSketch at matched budget, ~6 min
bash experiments/exp7_update_bench.sh      # update throughput, ~1 min
```

Every table script ends by printing a **facsimile** of the published table:
the reproduced value next to the published value, cell by cell, marked as a
match or a difference.  The archived log each number came from is bundled
under `results/` and can be re-checked independently.

## Using CCS as a library

Three lines, no data files, no paper context required:

```python
import numpy as np, ccs

sk = ccs.ComplexCountSketch(p=3, width=1024, depth=8, seed=1)
sk.update(stream)          # stream: iterable of keys (str / bytes / int)
print(sk.estimate())       # -> F_3 estimate
```

`update` is additive, so a stream can arrive in chunks and sketches built on
different machines merge without sharing the data (`sketch_a + sketch_b`).
One-call helpers cover the two headline use cases:

```python
ccs.fp_estimate(keys, counts, p=3, eps=0.1, gam=0.01, seed=1)
ccs.star_estimate(F, p=3, width=256, seed=1)      # k-way star join size
ccs.join2_estimate(R, S, p=4, width=1024, seed=1) # 2-way join size
sk.estimate("tmccs")                              # truncated-median variant
```

See `examples/README.md` for the full API and the measured output of each
demo.  `tests/test_api.py` pins this API to `ccs.cpp_ccs_estimate`
bit-for-bit, so the ergonomic wrapper is not a second implementation.

## Environment

- Reference: Python 3.11 + numpy 1.26.4 + matplotlib 3.10.0 (the ports were
  developed and verified in a `conda` env named `py311`; any Python 3.9+
  interpreter with numpy >= 1.17 works, only the bit-exactness claims below
  depend on the exact numpy version).
- RNG: everything uses `numpy.random.default_rng` (PCG64).  `Generator`'s
  integer draws and `choice(p=...)` are bit-identical across numpy
  1.17-1.26, so all "bit-exact" claims here were measured under numpy
  1.26.4.  **exp4 is the single exception**: its archived values were
  produced under an older numpy and the absolute values drift under the
  current one.  The paper only claims a *directional* conclusion from that
  experiment and that conclusion holds; see the disclosure in
  `experiments/exp4_equal_budget.py`.
- C++ needs `g++` (Apple Clang on macOS); the drivers depend only on the
  C++11/14 standard library.

## Data (data/, 409 MB)

| Content | Files | Used by |
|---|---|---|
| synthetic Lp columns | `lp/{uniform,zipf1.1,zipf1.5}.dat{,2,3}.gz`, `lp/uniform+zipf1.1.dat.gz` | Tables 2/3/4, exp9/exp13/exp18 |
| MovieLens | `ML/ratings/{userId,movieId,rating}.gz`, `ML/tags/{userId,movieId}.gz` | Tables 2/5, exp1/exp9/exp12/exp14/exp15 |
| Airlines | `airlines/*.gz` (10 columns) | Tables 2/5, exp15 |
| Kasandr | `kasandr_essential.npz` (users/cats/times), `kasandr_extra_columns.npz` (distinct views of offerid/countrycode/merchant), `kasandr_columns/*.kc.tsv.gz` (7-column merge views) | Tables 2/4/5, exp2/exp9/exp12/exp15 |

### Provenance and redistribution

The three real datasets are third-party.  This repository redistributes only
the **derived column projections** actually consumed by the scripts
(gzip-compressed columns and `.npz` views), not the datasets as published:

| Dataset | Source | Cite |
|---|---|---|
| MovieLens 25M | GroupLens Research, University of Minnesota | Harper & Konstan, *ACM TIIS* 5(4):19, 2015 |
| Airlines | OpenML dataset 42728 (Airlines Departure Delay) | <https://www.openml.org/d/42728> |
| Kasandr | Sidana, Laclau, Amini, Vandelle, Bois-Crettez — SIGIR 2017, pp. 1245–1248 | doi via the SIGIR 2017 proceedings |

Redistribution note: the bundled files are derived views, and no upstream
license files are shipped with them.  If you re-publish this repository,
check each dataset's own terms of use (MovieLens is distributed by GroupLens
for research use) and cite the dataset papers above alongside ours.

The synthetic Lp columns are committed **on purpose** and cannot be
regenerated: the original generator (`random.randint` / `np.random.zipf`,
10^7 draws per column) ran **unseeded**, so the bundled `.gz` files are the
only surviving record of those streams.  Re-seeding a generator now would
produce different data and different table numbers, so please do not
"optimise" them out of the repository — Tables 2 and 4 depend on them.

Every bundled file is **byte-identical** to the original used for the paper
(md5-checked; the nine Lp `.dat` files were compared one by one with `cmp`).
Kasandr's user/category columns are numbered by first appearance and `times`
is a YYYYMMDD integer, exactly as the original `load()` produced them.  The
plaintext datasets the C++ pipeline reads (~2 GB) are not bundled;
`experiments/exp5_join2_matched.sh` shows how to materialise one on the fly
from the bundled gzip columns.

## Reproduction status 

**Bit-exact**

| Object | Scope | Archived baseline |
|---|---|---|
| Table 2 | all 42 CCS cells match the published values; p=4/p=5 synthetic and every p on ML/Airlines/Kasandr match the C++ logs bit-for-bit | `results/table2/` |
| Table 3 | 83/84 cells (plain / T-MCCS / Fixed20 / Rand20 / G-A / wins) | `results/table3/` |
| Table 4 | after the fix below, 6 combinations x all columns and diagnostics | `results/table4/rev_pstar_full.txt` |
| Table 5 | 3-star rows + Count Sketch columns (T=24, full protocol) | `results/table5/` |
| Figures 1/2 | 15 + 36 recomputed data points all match the archives (3 significant digits) | `results/figures/`, `results/table3/` |
| exp2/3b/6/11/12/13/14/15/16/17/18 | full protocol | `results/experiments/` |
| exp5 (C++ JoinSketch) | bit-exact at matched budget | `results/experiments/exp5_join2_matched.expected.txt` |
| exp9 | the 6 synthetic combinations are covered by the Table 4 full-protocol run; the standalone full run was stopped at the 5th combination to free the machine for 
**Disclosed gaps and differences**

1. **Table 5, CCS p=3/4/5 columns.** The original run behind those columns
   was never archived, and neither of the two seed conventions we tried
   reproduces the published values.  The script reproduces the closest
   protocol and **labels the column as reference, not reproduction** (see the
   docstring of `tables/table5_realccs.py` and its facsimile output).
2. **Table 3, Fixed20 (p=3, alpha=1.0).** The paper says 2.00E-03; the
   archived log says 2.454e-03.  This package reproduces the archived value;
   the paper entry looks like a transcription slip, and the boldface
   conclusion in that row is unaffected.
3. **Table 2, AMS column.** The C++ pipeline draws its positions from
   `std::random_device`, so the AMS column is not reproducible by
   construction and takes no part in the cell-by-cell audit.  The CCS column
   is unaffected.
4. **exp4 absolute values.** Produced under an older numpy; they drift in the
   current environment.  The directional claim holds (the original producer
   only checked direction as well).
5. **exp7 throughput.** Machine dependent (original Linux server vs this
   machine); only the ordering CS > CCS > AMS is checked.
6. **A conjugation bug caught during the Table 4 port.** The first version of
   `table4_pstar.py` wrote the imaginary part of its inline complex product
   as `pr*ci + pi*ci` (correct: `pr*ci + pi*cr`).  The archived cross-check
   caught it on the spot; after the fix the port is bit-exact with the
   original producer `rev_pstar.py`.  This is the value of "get it running,
   then reconcile against the archive" as a standing rule.



## Subdirectory notes

- `tables/README.md`: porting details, audit conventions and known gaps for
  the table scripts.
- `experiments/README.md`: paper mapping and porting notes for the 16
  experiments.
- `cpp/README.md`: building the C++ pipeline, its data, and its audit
  conventions.
- `examples/README.md`: the standalone CCS API and the measured output of
  each demo.
