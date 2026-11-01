"""Exp 07 -- exact real multipliers at odd p (paper Section 4.2).

Solves the moment conditions EXACTLY on fixed supports (p=3 is the paper's
{-1,1,2} solution; p=5 and p=7 are found by exact linear solve on small
supports), then verifies:
  (a) odd moments and mu_p to machine precision;
  (b) unbiasedness of the power-sum estimator by exact expectation
      (dynamic program, no Monte Carlo);
  (c) the variance ratio of each real solution against the root-of-unity
      CCS multiplier on identical streams.
"""
import numpy as np
sys_path = None
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs


def exact_solve(p, sup):
    eqs = [np.ones(len(sup))] + [sup ** m for m in range(1, p, 2)] + [sup ** p]
    A = np.vstack(eqs); b = np.zeros(len(eqs)); b[0] = 1.0; b[-1] = 1.0
    w = np.linalg.solve(A, b)
    assert np.all(w > 1e-9), f"support {sup} not strictly feasible"
    return w


def variance_ratio(p, s, w, families=30, w64=64, d=64, D=200, N=50_000, seed=11):
    rng0 = np.random.default_rng(seed)
    freq = np.bincount(rng0.integers(0, D, size=N), minlength=D).astype(float)
    vr = []
    for t in range(families):
        rng = np.random.default_rng(1000 * p + t)
        rows = np.empty(d)
        for j in range(d):
            h = rng.integers(0, w64, size=D)
            k = rng.choice(len(s), size=D, p=w)
            C = np.bincount(h, weights=freq * s[k], minlength=w64)
            rows[j] = float((C ** p).sum())
        rng2 = np.random.default_rng(9000 * p + t)
        rows_c = np.empty(d)
        for j in range(d):
            h = rng2.integers(0, w64, size=D)
            k = rng2.integers(0, p, size=D)
            g = np.exp(2j * np.pi * k / p)
            Cr = np.bincount(h, weights=freq * g.real, minlength=w64)
            Ci = np.bincount(h, weights=freq * g.imag, minlength=w64)
            rows_c[j] = float(((Cr + 1j * Ci) ** p).real.sum())
        vr.append(np.var(rows) / np.var(rows_c))
    return float(np.median(vr))


if __name__ == "__main__":
    sols = {3: (np.array([-1.0, 1.0, 2.0]), np.array([7 / 12, 1 / 4, 1 / 6]))}
    sup5 = np.array([-1.0, 0.5, 1.5, 2.5])
    sup7 = np.array([-2.0, -1.5, -1.0, 1.5, 4.0])   # mildest feasible support
    sols[5] = (sup5, exact_solve(5, sup5))
    sols[7] = (sup7, exact_solve(7, sup7))

    rng0 = np.random.default_rng(11)
    freq = np.bincount(rng0.integers(0, 200, size=50_000), minlength=200).astype(float)

    for p, (s, w) in sorted(sols.items()):
        worst = max(abs(float((w * s ** m).sum())) for m in range(1, p, 2))
        mu_p = float((w * s ** p).sum())
        mu = ccs.g_moments(s, w, p)
        E = ccs.exact_power_expectation(p, freq, mu)
        truth = float((freq ** p).sum())
        ratio = variance_ratio(p, s, w)
        print(f"p={p}: support={s.tolist()}")
        print(f"   weights={[round(float(x), 8) for x in w.tolist()]}")
        print(f"   worst |odd moment| = {worst:.2e}   mu_p - 1 = {mu_p - 1:.2e}")
        print(f"   exact E[sum C^p] / F_p - 1 = {E / truth - 1:.2e}")
        print(f"   variance ratio real/CCS = {ratio:.1f}")
    print("EXP-REAL-SOL DONE")
