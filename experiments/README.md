# experiments/ — the paper's in-text experiments

Self-contained reproductions of the experiments reported **in the text** (not
in a table).  The table scripts live in `../tables/` (see that README); this
directory covers everything else.  All scripts depend only on `numpy`
(+ `matplotlib`, figures only) plus the repository's `ccs.py` / `loaders.py`,
and read all data from `../data/` (gzip column files, byte-identical to the
paper's originals).

```bash
cd Complex_Count_Sketch
python3 experiments/run_all.py --list      # the steps and their expected cost
python3 experiments/run_all.py             # run everything, then audit (~2 h)
python3 experiments/run_all.py exp11 exp16 # run only the chosen steps
python3 experiments/run_all.py --verify    # audit existing results/*.out only
```

Reference environment: Python 3.11 + numpy 1.26.4 (see "Environment" in the
root README).  Output goes to `../results/<step>.out` and is audited against
the archived production logs in `../results/experiments/`.

## Script -> paper location

| Script | Paper location | Content | Audit result |
|---|---|---|---|
| `exp2_kasandr_3star.py` | §5.5 | real Kasandr 3-star: time-window shards (aligned) vs top-3 categories (misaligned) | bit-exact |
| `exp3b_aggregators.py` | §5.3 "Against other aggregators" | four row aggregators on one sketch: median / mean / median-of-means / T-MCCS | bit-exact (21/21 configs) |
| `exp4_equal_budget.py` | §5.2 | CCS contracted to the AMS budget (w=100, d=6) on synthetic data | directional claim holds; absolute values drift with the numpy version (disclosed in the script) |
| `exp5_join2_matched.sh` | Table 5 `tab:realccs` JoinSketch column (matched-budget reference) | JoinSketch at matched budget (131,072 counters), C++ build + run | needs g++; data materialised on the fly from the bundled gz; archived values reflect the old key interface (disclosed in the README) |
| `exp6_ganguly_compare.py` | §5.2 "Comparison with Ganguly's estimator" | comparison against Ganguly (2004) at equal space | bit-exact |
| `exp7_update_bench.sh` | §5.1 Hardware | update-throughput benchmark (C++, 10^6 insertions) | machine dependent; only the ordering is checked |
| `exp8_fp_ci_sweep.py` | §5.2 CI + width sweep | 95% CIs for F_p (30 repetitions) + width sweep | part (b) is verified by the full recomputation in `../tables/figures.py` (15/15) |
| `exp9_pstar_baselines.py` | §5.4 (Table 4 baselines) + §5.5 | real-valued p-star baselines: per-edge (unbiased) / shared-sign / cubes; 6 synthetic combinations + ML + Kasandr | the 6 synthetic combinations are bit-exact with the full-protocol run of `../tables/table4_pstar.py`; in the standalone full run the first 4 constructs were confirmed bit-exact before it was stopped to free the machine for the notebook; because the archived tags are truncated, `run_all.py` reconciles by **ground truth** |
| `exp11_pattern_check.py` | §4.5 + §5.2 checklist | exact enumeration of co-location patterns: counterexample formula, ½Mix², three-item coefficient | bit-exact |
| `exp12_real_multistar.py` | §5.5 "Five-table stars" | real 5-table star (p=5): 5 ML shards + Kasandr top-5 categories | bit-exact |
| `exp13_misalignment_sweep.py` | §5.4 "Misalignment, not skew" | fixed marginals, permute a fraction of the heavy keys: separates skew from support misalignment | bit-exact (10/10 configs) |
| `exp14_merge_delete.py` | §5.6 | 16-site sharded merge: estimate difference, communication volume (134x), delete unbiasedness | bit-exact |
| `exp15_planner_rank.py` | §5.7 "From Estimates to Planning" | F_3 ranking over 10 Airlines columns / 6 Kasandr fields: Spearman + top-3 | bit-exact (Kasandr string columns use crc32 keys; disclosed in the script) |
| `exp16_real_solutions.py` | §4.2 real multipliers | exact solve of the odd-order real solution + exact expectation unbiasedness + variance ratios (13.8 / 2.6 / 9097) | bit-exact |
| `exp17_ganguly_attribution.py` | §5.2 Ganguly comparison | equal-memory 2x2 attribution: allocation x aggregation | bit-exact |
| `exp18_joint_workload.py` | §5.5 "One deployment, three tasks" | one set of CCS sketches answers F_3 / 2-join / 3-star (0.79 MB vs 30.5 MB) | bit-exact |

## Experiments covered by tables/ (not duplicated here)

| Original script | Now lives in | Note |
|---|---|---|
| `exp1_real_joins.py` | `../tables/table5_realccs.py` | Table 5: real 3-star + real 2-join (p=2..5 vs Count Sketch) |
| `exp3_tmccs_stats.py` | `../tables/table3_tmccs.py` | Table 3: CCS vs T-MCCS + CIs + sign test |
| `exp8_fp_ci_sweep.py` (b) | `../tables/figures.py` | Figure 1 (width sweep): full recomputation and archive check |
| `rev_pstar.py` | `../tables/table4_pstar.py` | Table 4 CCS/T-MCCS columns and diagnostics |
| exp9, synthetic part | `../tables/table4_pstar.py` | same baseline code path |

## Differences from the original scripts (porting notes)

1. **Data layer.** The originals read plaintext column files under
   `experiment/.../datasets/` via `common.py`; the port reads the bundled
   gzip through `../loaders.py` (byte-identical content).  The deprecated
   `np.fromstring` calls were replaced with equivalent `loaders` parsing.
2. **Estimator layer.** Each script's inline complex multiplication and
   bucket accumulation were replaced with the verified functions in
   `../ccs.py` (`star_rows` / `ccs_rows_batch` / `peredge_star` /
   `sharedsign_star` / `cubes_real` / `tmccs` / `median`, ...).  RNG draw
   order and seeds were compared against the original producers one by one;
   results are bit-exact (table above).
3. **exp15 key function.** The original hashed Kasandr string columns with
   Python's built-in `hash()`, which is salted per process (PYTHONHASHSEED)
   and therefore not reproducible; the port uses `zlib.crc32` instead.  The
   F_3 ground truth and the CCS/cubes estimates depend only on distinct counts
   and are unaffected; the key order of the sampled base columns differs but
   their ranking metrics do not (checked by running both; all metrics agree).
4. **exp4 absolute values.** The script was ported verbatim, but the archived
   log was produced under an older numpy and the absolute values drift in the
   current environment (the original producer only checked direction as
   well).  The paper claims only a directional conclusion from this
   experiment, and it holds.
5. **exp5/exp7 (C++).** Sources are in `../cpp/`.  `exp5` materialises the 8
   plaintext column files the drivers need from the bundled gz at run time
   (~450 MB, deleted afterwards; `--keep` retains them).
6. **Truncated tags in the exp9 archives.** When the logs were captured the
   combination names were truncated (`uniforxzipf1.xzipf1.`; zipf1.1 and
   zipf1.5 collide after truncation, so synth2/synth3 share a tag and so do
   synth4/synth5).  `run_all.py` therefore reconciles exp9 by **ground truth**
   (unique per construct, printed identically on both sides) rather than by
   tag or by whole-line matching.
7. **Runner bookkeeping lines in the archives.** The original producer's
   run_all appended an exit marker after each step, in two formats
   (`EXP6_EXIT 0` and `E12 0`); the scripts themselves never print them and
   the audit ignores them.  The exp5 archive also contains that machine's g++
   warnings (different absolute paths), so its audit compares only the
   driver's result lines (`userId,` / `movieId,` / `result:`).

## Data

- MovieLens (25M rating rows + 1.09M tag rows), Airlines (10M rows, 10
  columns), Kasandr (15.84M rows) and the synthetic Lp columns (10^7 rows
  each) are all bundled under `../data/`.
- Kasandr's user/category columns are numbered **by first appearance** (as the
  original `load()` did) and `times` is a YYYYMMDD integer.  The three extra
  columns exp15 needs (offerid / countrycode / merchant) are stored as
  distinct views in `data/kasandr_extra_columns.npz`.
