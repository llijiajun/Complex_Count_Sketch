# tables/ — reproduction scripts for the paper's tables and figures

Each script is a **self-contained reproduction** of one table or figure: a
faithful port of the original producer (RNG draw order, seed base and protocol
parameters preserved one by one) that finishes by printing a **facsimile** of
the published table — reproduced value next to published value, cell by cell,
each marked as a match or a difference.  The archived production logs are
bundled under `../results/table{2,3,4,5}/` and `../results/figures/`.

```bash
python3 tables/table2_lpnorm.py       # Table 2   (~8 min)
python3 tables/table3_tmccs.py        # Table 3   (~4 min)
python3 tables/table4_pstar.py        # Table 4   (~45 min)
python3 tables/table5_realccs.py      # Table 5   (~6 min)
python3 tables/figures.py             # Figures 1/2 from the archives (seconds)
python3 tables/figures.py --recompute # recompute + verify (~2 min)
```

## Script -> paper

| Script | Paper | Protocol | Audit status (measured 2026-10-23) |
|---|---|---|---|
| `table2_lpnorm.py` | Table 2 `table:lpnorm` | 4 synthetic distributions + ML/Airlines/Kasandr x p=3,4,5, mean/median, 5 rounds | **all 42 CCS cells match the published values**; p=4/p=5 synthetic and every p on the three real datasets match the C++ archives bit-for-bit; no p=3 synthetic archive exists (reproduced by the port; `result.csv` was shown to be stale) |
| `table2_archive_audit.py` | — | automatic check of every Table 2 cell against `results/table2/` | all OK |
| `table3_tmccs.py` | Table 3 `table:tmccs` | alpha in {0,.5,.8,1,1.1,1.5,2} x p=3,4,5, 1000 runs | **83/84 bit-exact**; the single difference is Fixed20(p=3, alpha=1.0): paper 2.00E-03 vs archive 2.454e-03 (looks like a transcription slip; the boldface conclusion is unchanged) |
| `table4_pstar.py` | Table 4 `table:pstar` | p=3, w=256, d=64, T=100; rev_pstar protocol + exp9 baselines | after the conjugation fix, **every column matches** (below) |
| `table5_realccs.py` | Table 5 `tab:realccs` | real 3-star (T=24) + 2-join at orders p=3,4,5 + Count Sketch w=2048 | 3-star and Count Sketch columns bit-exact; **the CCS column is a disclosed provenance gap** (below) |
| `figures.py` | Figures 1/2 | fig_sweep = exp8b protocol; fig_regime = exp3 data | `--recompute` re-derives 15 + 36 points, all match at 3 significant digits |

## Two gaps you must know about (on the paper's side, not the port's)

1. **Table 5, CCS p=3/4/5 columns.** The run behind those columns was never
   archived, and neither seed convention we tried (4000+p and 4000+100p)
   matches the published values (e.g. AL Origin-Dest p=3: paper 2.5E-03,
   archived protocol 2.97E-03).  `table5_realccs.py` reproduces the closest
   protocol and **labels the column as reference** in its facsimile.  The
   JoinSketch column is likewise an external C++ baseline, quoted as a
   reference value; its matched-budget reproduction lives in
   `experiments/exp5` and is bit-exact.
2. **Table 2, AMS column.** The C++ pipeline draws its positions from
   `std::random_device`, so it is not reproducible; the paper's AMS column
   takes no part in the cell-by-cell audit.  The CCS column is unaffected.

## The Table 4 conjugation bug (evidence that the process works)

The first version of `table4_pstar.py` wrote the imaginary part of its inline
complex product as `pr*ci + pi*ci` (correct: `pr*ci + pi*cr`), which
systematically skewed the CCS/T-MCCS columns (uuu: reproduced 2.71E-01 vs
published 1.03E-01).  The archived cross-check caught it immediately.  After
the fix the port is bit-exact with the original producer `rev_pstar.py` under
py311 (the single uuu combination already confirms all ten quantities
A/G/B/D/G-A/D-A/win/Asd/Gsd/corr), and the full protocol over all 6
combinations then matches `results/table4/rev_pstar_full.txt` and the paper's
Table 4.  The lesson — **every table is reconciled against its archive before
it ships** — is the acceptance criterion for every number in this package.

## Porting layer

- The table scripts never read plaintext data directly; everything goes
  through `../loaders.py` (bundled gzip, byte-identical to the originals).
  Columns whose semantics come from the C++ hash (ML/Airlines/Kasandr in
  Table 2) use the `*_lines` accessors to obtain the raw line-string view that
  the C++ drivers hash.
- Complex multiplication, bucket accumulation and the T-MCCS truncation all
  come from `../ccs.py`, where each function was compared against and
  measured against its original producer.
- Number formatting follows the paper: Table 2 uses 2-digit-mantissa E
  notation and the audit compares strings exactly; Table 5 uses 2 significant
  digits.
