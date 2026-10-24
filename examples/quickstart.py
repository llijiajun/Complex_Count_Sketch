"""Quickstart: estimate the p-th frequency moment F_p of a stream with CCS.

Run from the artifact root (or anywhere -- the script finds the root itself):

    python3 examples/quickstart.py

The whole script is three lines of CCS plus bookkeeping:

    sk = ccs.ComplexCountSketch(p=3, width=1024, depth=8, seed=1)
    sk.update(stream)
    print(sk.estimate())

It then shows what the knobs do: error shrinks like 1/sqrt(width), the sketch
is a fixed-size summary no matter how long the stream is, and two sketches of
two substreams can be merged instead of re-reading the data.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ccs  # noqa: E402

D, N, ALPHA = 50_000, 1_000_000, 1.1     # keys, stream length, Zipf skew
P = 3


def main():
    # ---------------------------------------------------------------- stream
    rng = np.random.default_rng(0)
    w = np.arange(1, D + 1, dtype=np.float64) ** (-ALPHA)
    w /= w.sum()
    stream = rng.choice(D, size=N, p=w)          # the "data": one key per row

    # ------------------------------------------------------------- the truth
    freq = np.bincount(stream, minlength=D).astype(np.float64)
    truth = float((freq ** P).sum())
    print(f"stream: N={N:,} rows over D={D:,} keys, Zipf(alpha={ALPHA})")
    print(f"exact  F_{P} = {truth:.6e}\n")

    # ------------------------------------------------------- 1. the sketch ---
    # A (depth x width) array of complex counters -- 8 x 1024 complex values
    # summarise a 10^6-row stream.  Nothing about the sketch depends on N.
    t0 = time.time()
    sk = ccs.ComplexCountSketch(p=P, width=1024, depth=8, seed=1)
    sk.update(stream)
    est = sk.estimate()
    dt = time.time() - t0
    print(f"CCS    F_{P} = {est:.6e}   "
          f"(rel. err {abs(est - truth) / truth:7.3%}, "
          f"{sk.memory_bytes() / 1024:.0f} KiB of counters, {dt:.2f}s)")

    # -------------------------------------------- 2. width controls accuracy --
    print("\nwidth sweep (error ~ 1/sqrt(width); depth=8 fixed):")
    print(f"  {'width':>7}  {'estimate':>14}  {'rel. err':>9}")
    for width in (64, 256, 1024, 4096, 16384):
        s = ccs.ComplexCountSketch(p=P, width=width, depth=8, seed=1)
        s.update(stream)
        e = s.estimate()
        print(f"  {width:>7}  {e:>14.6e}  {abs(e - truth) / truth:>8.3%}")

    # ----------------------------------------- 3. other orders, one sketch ----
    # p >= 3 is where complex roots of unity are needed; p = 2 degenerates to
    # the classical (real) Count Sketch, which CCS also reproduces.
    print("\nother orders (width=1024, depth=8):")
    print(f"  {'p':>2}  {'estimate':>16}  {'rel. err':>9}")
    for p in (2, 3, 4, 5):
        s = ccs.ComplexCountSketch(p=p, width=1024, depth=8, seed=1)
        s.update(stream)
        t = float((freq ** p).sum())
        e = s.estimate()
        print(f"  {p:>2}  {e:>16.6e}  {abs(e - t) / t:>8.3%}")

    # ------------------------------------------- 4. the stream, in chunks -----
    # update() is additive, so a stream can arrive in pieces ...
    a = ccs.ComplexCountSketch(p=P, width=1024, depth=8, seed=1)
    b = ccs.ComplexCountSketch(p=P, width=1024, depth=8, seed=1)
    a.update(stream[: N // 2])
    b.update(stream[N // 2:])

    # ... and two parties can merge their sketches without sharing the data.
    merged = a + b
    e = merged.estimate()
    print(f"\nmerged two half-stream sketches: F_{P} = {e:.6e}   "
          f"(rel. err {abs(e - truth) / truth:7.3%})")

    # ------------------------------------------ 5. one-call convenience API ---
    keys, counts = np.unique(stream, return_counts=True)
    one = ccs.fp_estimate([str(k) for k in keys], counts, p=P,
                          eps=0.1, gam=0.01, seed=1)
    print(f"fp_estimate(eps=0.1, gam=0.01): F_{P} = {one:.6e}   "
          f"(rel. err {abs(one - truth) / truth:7.3%})")


if __name__ == "__main__":
    main()
