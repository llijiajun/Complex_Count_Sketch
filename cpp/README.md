# cpp/ — the original C++ pipeline (2024, Linux x86 server)

The original C++ pipeline behind the paper's **real-data tables** (currently
numbered Table 2, plus the reference column of Table 5).  The reorganised
Python ports (`../tables/`, `../experiments/`) cover every experiment in the
paper; this directory preserves the historical code path, and two of its
experiments — the JoinSketch baseline (exp5) and the update-throughput
benchmark (exp7) — can be re-run directly from the artifact.

Note on labels: the table numbers and labels in each `.cpp` header comment are
the **2024-era** ones (`table:sylpnorm`, `table:realpnorm`, `table:synjs`,
...).  The submitted paper merged and renamed them: synthetic and real l_p
data are now one **Table 2 (`table:lpnorm`)**, the synthetic 2-join comparison
moved into the text of §5.3 (no table), and the real 2-join at higher orders
is **Table 5 (`tab:realccs`)**.  The mapping below uses the **current** paper.

## 1. Building

Every driver depends only on the C++11/14 standard library:

```bash
cd cpp
g++ -O2 -std=c++11 -I common -I common/include join2/join2.cpp -o join2
```

All `.cpp` files in the package compile under Apple Clang 17 (arm64) and g++
(on arm64 `g++` is Apple Clang; Linux g++ also passes).  Compared with the
2024 original, `common/countsketch.hpp` carries three portability patches
(each marked with a `paper_code patch` comment): conditional inclusion of the
x86 intrinsics, unconditional compilation of the definitions inside the
`#ifdef UNIX` block, and a scalar fallback for `SIMD_match_8` (bit-identical
results).

## 2. File -> paper table

| File | Paper location | Note |
|---|---|---|
| `common/countsketch.hpp` | §2–§3 sketch structure | AMS / CountSketch / **ComplexCountSketch** / MGCS / Classifier (JoinSketch) |
| `common/MurmurHash.h` | §5.1 Setup | 32-bit MurmurHash, the hash function the paper declares |
| `lp_norm/main.cpp` | **Table 2** `table:lpnorm` (synthetic block) | synthetic l_p norm (AMS vs CCS); `int p=` selects the 3/4/5 column groups |
| `lp_norm/ML.cpp` | **Table 2** `table:lpnorm`, MovieLens rows | ratings.csv + tags.csv |
| `lp_norm/Airlines.cpp` | Table 2, Airlines rows | airlines arff |
| `lp_norm/kasandr.cpp` | Table 2, Kasandr rows | kasandr.zip |
| `lp_norm/SSB.cpp` | none (historical) | SSB is not used by the paper |
| `lp_norm/MG4CCS.cpp` | none (historical) | the MG family was removed from the paper; kept as archive |
| `join2/join2.cpp` | §5.3 text (synthetic 2-join comparison; no table in the paper) | synthetic 2-join: CountSketch column + JoinSketch (Classifier) column |
| `join2/MLjoin2_fixed.cpp` | **Table 5** `tab:realccs`, JoinSketch column | MovieLens JoinSketch after the key-interface fix (MurmurHash32 of the whole line) + budget sweep |
| `join2/MLjoin2_matched.cpp` | historical (old key interface) | matched-budget JoinSketch on MovieLens; the archived values reflect the old interface (truncated userId) |
| `join2/ALjoin2_matched.cpp` | historical (old key interface) | matched-budget JoinSketch on Airlines |
| `join2/{Airlines,Kasandr,ML}join2.cpp` | none (historical) | the real-data 2-join table they fed was deleted from the paper |
| `multijoin3/*` | none (historical) | real 3-join and timing; its `countsketch.hpp` is a compact reference implementation of CCS |
| `bench/bench_update.cpp` | §5.1 Hardware | update-throughput benchmark (CCS / Count Sketch / AMS, 10^6 insertions) |
| `datasets_tools/*.py` | Table 2 / §5.3 ground truth | data preprocessing and join ground-truth statistics |

## 3. The two experiments you can re-run

```bash
cd Complex_Count_Sketch
bash experiments/exp5_join2_matched.sh   # JoinSketch at matched budget (~6 min, needs g++)
bash experiments/exp7_update_bench.sh    # update throughput (~1 min, needs g++)
```

- **exp5**: the drivers read plaintext column files through the relative path
  `../datasets/...`.  The script first materialises the 8 files it needs from
  the bundled `../data/*.gz` into `cpp/join2/datasets/` (~450 MB, deleted
  afterwards; `--keep` retains them), then builds and runs.  Measured
  **bit-exact** against the archived log (the `userId` line
  1.36533e+09 / 1.3264e+10 / 1.28485e+09 and the rest all match).  Note that
  the archived values reflect the **old key interface** of
  `MLjoin2_matched.cpp` (`Classifier::Insert(const char*)` reads only the
  first 4 bytes of the line, so userId is truncated and the Count Sketch
  column is systematically biased high).  That behaviour is deterministic and
  therefore reproducible; the version with the corrected key interface is
  `MLjoin2_fixed.cpp`, see also `join2/run_matched/ML2join_fixed.txt`.
- **exp7**: a purely synthetic stream, no data dependency.  Absolute
  throughput varies by machine (the original Linux server achieved 72.5M/s for
  Count Sketch against 34.0M/s here); the ordering CS > CCS > AMS is stable.

## 4. Data

The **original plaintext datasets** needed by `lp_norm/*.cpp` and
`join2/*.cpp` (ratings.csv, the airlines arff, kasandr.zip, the synthetic Lp
columns) total about 2 GB and are not distributed with the package; they live
in the original repository under `experiment/Distributed JoinSize/datasets/`.
The reorganised artifact bundles their column views as `../data/*.gz`
(byte-identical to the originals, md5-checked), and every Python reproduction
script reads those.  To run a C++ driver, either materialise the files from
the bundled gz (as the exp5 script does) or symlink the original repository's
datasets directory to `cpp/lp_norm/../datasets` and `cpp/join2/datasets`.

## 5. How the numbers reconcile with the paper (stated honestly)

1. **The Table 2 numbers in the paper are averages over several runs (5-10
   rounds)**, post-processed; the raw single-run outputs and the per-column
   xlsx summaries are in the original repository's `Lp/` directory.  The
   bit-exactness convention on the Python side is documented in
   `../tables/README.md` (all 42 CCS cells and the p=4/p=5 synthetic archives
   match bit-for-bit; no p=3 synthetic archive survives, so the port supplies
   those numbers).
2. **The JoinSketch numbers of the §5.3 synthetic 2-join comparison** come
   from multi-round averaging in the same way; `datasets_tools/true.py` and
   friends produce the ground truth.
3. **The JoinSketch column of Table 5 (`tab:realccs`)** is an external C++
   baseline at matched budget (corrected key interface, see
   `join2/MLjoin2_fixed.cpp`); its CCS p=3/4/5 column has a known provenance
   gap (the paper's run was never archived), disclosed in
   `../tables/README.md` and in the docstring of `table5_realccs.py`.
4. `lp_norm/main.cpp` and `lp_norm/ML.cpp` also contain MGCCS columns, a
   method the paper dropped; ignore them when reconciling.
5. The 2024 `.o` binaries were built against older headers and are not
   distributed.  A few `.cpp` files had drifted against the current headers;
   the package patches them (marked comments), which only touches pointer /
   reference parameters and one unused variable, not the algorithms.
