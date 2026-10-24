# examples/ — use CCS as a library

These four scripts are the short path for a reader who wants to **use** CCS,
not reproduce the paper's tables.  They depend only on `ccs.py` (no archived
logs, no bundled datasets), run in seconds, and print exact-vs-estimated
numbers you can sanity-check immediately.

```bash
cd Complex_Count_Sketch
python3 examples/quickstart.py       # ~3 s   F_p of a stream, in 3 lines
python3 examples/join_demo.py        # ~30 s  3-way star and 2-way joins
python3 examples/tmccs_demo.py       # ~2 s   when to truncate on |Im|
python3 examples/one_deployment.py   # ~2 s   one sketch, three tasks
```

Then the paper's tables and the other 16 experiments live in `../tables/` and
`../experiments/`; the reconciliation notebook is `../notebooks/`.

## The API in three lines

```python
import numpy as np, ccs

sk = ccs.ComplexCountSketch(p=3, width=1024, depth=8, seed=1)
sk.update(stream)          # stream: an iterable of keys (str/bytes/int)
print(sk.estimate())       # -> F_3 estimate
```

`update` is additive, so a stream can arrive in chunks, and two sketches built
on different machines can be combined without sharing the data:

```python
whole = sketch_a + sketch_b          # == a sketch of the union
```

Other entry points:

| call | what it answers |
|---|---|
| `ccs.fp_estimate(keys, counts, p, eps, gam, seed)` | F_p with the paper's (eps, gam) sizing |
| `ccs.star_estimate(F, p, width, seed, depth)` | k-way star join size, `F` = list of frequency vectors |
| `ccs.join2_estimate(R, S, p, width, seed, depth)` | 2-way join size, `R`/`S` = `(keys, counts)` |
| `sk.estimate("tmccs")` | same sketch, truncated-median aggregation |
| `sk.row_values()` | per-row `(Re, |Im|)` if you want your own aggregator |
| `sk.counters()`, `sk.memory_bytes()` | raw `(CR, CI)` arrays and their footprint |

`tests/test_api.py` pins this API to the paper's own estimator bit-for-bit, so
the ergonomic wrapper is not a second implementation.

## What each script shows

### `quickstart.py` — F_p of a stream

1M-row Zipf(1.1) stream over 50k keys, exact F_3 = 3.101e+15:

```
CCS    F_3 = 3.098783e+15   (rel. err  0.080%, 128 KiB of counters, 2.51s)

    width        estimate   rel. err
       64    3.060980e+15    1.299%
      256    3.073571e+15    0.893%
     1024    3.098783e+15    0.080%
     4096    3.099314e+15    0.063%
    16384    3.101001e+15    0.009%
```

Three things to take away: the sketch is a fixed-size summary regardless of
stream length; error falls as `width` grows with `depth` fixed; and p = 2, 3,
4, 5 all work from the same object (p = 2 degenerates to the classical real
Count Sketch).

### `join_demo.py` — multi-way and 2-way join sizes

3-way star over 200k shared keys, and a 2-way join, against the real-valued
baselines:

```
  method                              mean rel. err
  CCS (p=3, complex roots)                  0.267%
  CCS + T-MCCS truncation                   0.145%
  per-edge Rademacher (byte-matched)        0.133%
  shared sign (biased, 2x wider)          112.752%
```

Read this as the paper does: **at equal bytes the per-edge real sketch is the
stronger star estimator** — that is the honest reading of Table 4.  What the
shared complex root keeps is one hash per key instead of p-1, the exact
collision law behind the misalignment diagnosis, the imaginary-part signal
T-MCCS trades on, and one deployment that also answers F_p.  The *biased* real
constructions (shared sign, cubes) are not competitive at any width: their
median estimate is negative in about half the hash families.

### `tmccs_demo.py` — when truncation pays

```
   alpha  plain median     T-MCCS   ratio  corr(|Im|,err)
     0.0       3.769%    3.888%    1.03            0.02
     0.5       2.695%    2.705%    1.00            0.16
     0.8       0.889%    0.731%    0.82            0.48  <- T-MCCS helps
     1.0       0.361%    0.255%    0.71            0.56  <- T-MCCS helps
     1.5       0.031%    0.019%    0.61            0.70  <- T-MCCS helps
     2.0       0.002%    0.001%    0.49            0.67  <- T-MCCS helps
```

The gain tracks the collision signal in the last column, not the skew alone:
T-MCCS is an enhancement for skewed inputs, and a tie on uniform ones.

### `one_deployment.py` — one sketch, three tasks

Three Zipf columns sketched once with a shared (h, g), then queried three
ways from the same counters:

```
one deployment: 3 x 64 x 256 complex counters = 0.75 MiB
                (vs 22.9 MiB of exact key-domain arrays, 31x)

  task                           estimate            exact  rel. err
  F_3 of column 1            2.197126e+15     2.195385e+15   0.079%
  2-join (cols 1,2)          2.284299e+10     2.284237e+10   0.003%
  3-star (cols 1,2,3)        2.180642e+15     2.178898e+15   0.080%
```

The exponents (+1,-1) for the join and (+1,+1,+1) for the star both sum to 0
mod p, which is why one stored sketch serves every task.  No real-valued
multiplier inside the power-sum family is unbiased for F_3, so a real toolkit
covering the same three tasks needs the exact arrays — hence the 31x.
