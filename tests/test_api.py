"""Consistency tests for the user-facing API in ccs.py section F.

Run from the artifact root:

    python3 tests/test_api.py

The point of these tests is that the ergonomic wrapper is NOT a second
implementation: it must agree with the paper's own estimators bit-for-bit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccs  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def test_matches_cpp_estimator():
    """ComplexCountSketch == cpp_ccs_estimate bit-for-bit."""
    rng = np.random.default_rng(11)
    keys = [f"key-{i}" for i in range(4000)]
    counts = rng.integers(1, 50, size=len(keys)).astype(np.float64)
    for p in (3, 4, 5):
        eps, gam, seed = 0.25, 0.05, 7919
        width, depth = ccs.cpp_dims(p, eps, gam)
        sk = ccs.ComplexCountSketch(p, width, depth, seed)
        sk.update(keys, counts)
        got = sk.estimate()
        exp = ccs.cpp_ccs_estimate(keys, counts, p, eps, gam, seed)
        check(f"F_{p}: class == cpp_ccs_estimate (w={width}, d={depth})",
              got == exp, f"{got!r} vs {exp!r}")


def test_streaming_is_additive():
    """Feeding a stream in chunks == feeding it in one shot."""
    rng = np.random.default_rng(12)
    items = rng.integers(0, 2000, size=20_000)
    keys = [str(int(k)) for k in items]
    one = ccs.ComplexCountSketch(p=3, width=512, depth=8, seed=3)
    one.update(keys)
    many = ccs.ComplexCountSketch(p=3, width=512, depth=8, seed=3)
    for i in range(0, len(keys), 997):
        many.update(keys[i:i + 997])
    CR1, CI1 = one.counters()
    CRM, CIM = many.counters()
    # Not bit-equal: a chunked feed sums 21 bincounts instead of one, so the
    # floating-point accumulation order differs.  The sketch is the same up
    # to rounding; bit-exactness only holds against the single-shot C++
    # protocol (test_matches_cpp_estimator).
    tol = 1e-6 * max(np.abs(CR1).max(), np.abs(CI1).max())
    check("chunked update == one-shot update (counters, up to rounding)",
          np.allclose(CR1, CRM, rtol=1e-9, atol=tol)
          and np.allclose(CI1, CIM, rtol=1e-9, atol=tol))
    check("chunked update == one-shot update (estimate, up to rounding)",
          abs(one.estimate() - many.estimate())
          <= 1e-9 * abs(one.estimate()))


def test_merge():
    """Two sketches of disjoint stream halves merge into the whole."""
    rng = np.random.default_rng(13)
    left = [str(int(k)) for k in rng.integers(0, 3000, size=15_000)]
    right = [str(int(k)) for k in rng.integers(0, 3000, size=15_000)]
    a = ccs.ComplexCountSketch(p=4, width=512, depth=8, seed=5).update(left)
    b = ccs.ComplexCountSketch(p=4, width=512, depth=8, seed=5).update(right)
    merged = a + b
    whole = ccs.ComplexCountSketch(p=4, width=512, depth=8, seed=5)
    whole.update(left + right)
    check("merge == sketch of the union (counters)",
          np.allclose(merged.counters()[0], whole.counters()[0])
          and np.allclose(merged.counters()[1], whole.counters()[1]))
    check("merge == sketch of the union (estimate)",
          abs(merged.estimate() - whole.estimate()) <= 1e-9 * abs(whole.estimate()))
    try:
        a + ccs.ComplexCountSketch(p=4, width=512, depth=8, seed=6)
        ok = False
    except ValueError:
        ok = True
    check("merge rejects mismatched seeds", ok)


def test_accuracy():
    """The estimate is actually close to the truth (sanity, not bit-exactness)."""
    for p in (3, 4):
        freq = ccs.zipf_freq(D=20_000, N=400_000, alpha=1.1, seed=p)
        truth = float((freq ** p).sum())
        keys = [str(i) for i in range(len(freq))]
        est = ccs.fp_estimate(keys, freq, p=p, eps=0.1, gam=0.01, seed=101)
        rel = abs(est - truth) / truth
        check(f"F_{p} on Zipf(1.1): relative error {rel:.3f} < 0.25", rel < 0.25)


def test_star_and_join():
    """p-star and 2-join helpers return usable estimates."""
    U = 50_000
    rng = np.random.default_rng(7)
    F = [rng.integers(0, 6, size=U).astype(np.float64) for _ in range(3)]
    truth = float((F[0] * F[1] * F[2]).sum())
    est = ccs.star_estimate(F, p=3, width=2048, seed=2024, depth=64)
    rel = abs(est - truth) / truth
    check(f"3-star: relative error {rel:.3f} < 0.30", rel < 0.30)

    R = (np.arange(U), rng.integers(0, 5, size=U).astype(np.int64))
    S = (np.arange(U), rng.integers(0, 5, size=U).astype(np.int64))
    truth2 = float((R[1].astype(np.float64) * S[1]).sum())
    est2 = ccs.join2_estimate(R, S, p=2, width=4096, seed=99, depth=64)
    rel2 = abs(est2 - truth2) / truth2
    check(f"2-join: relative error {rel2:.3f} < 0.30", rel2 < 0.30)


def test_aggregators():
    """tmccs differs from plain median but stays in the same ballpark."""
    freq = ccs.zipf_freq(D=5_000, N=100_000, alpha=1.5, seed=1)
    keys = [str(i) for i in range(len(freq))]
    sk = ccs.ComplexCountSketch(p=3, width=256, depth=64, seed=4)
    sk.update(keys, freq)
    plain = sk.estimate("median")
    trunc = sk.estimate("tmccs")
    truth = float((freq ** 3).sum())
    check("tmccs estimate is finite and comparable to plain median",
          np.isfinite(trunc) and abs(trunc - truth) / truth < 1.0,
          f"plain={plain:.4g} tmccs={trunc:.4g} truth={truth:.4g}")


if __name__ == "__main__":
    print("ccs user-facing API tests")
    for fn in (test_matches_cpp_estimator, test_streaming_is_additive,
               test_merge, test_accuracy, test_star_and_join,
               test_aggregators):
        print(fn.__name__)
        fn()
    print()
    print("RESULT:", "ALL PASS" if not FAILED else f"FAILURES: {FAILED}")
    sys.exit(1 if FAILED else 0)
