r"""Experiment 4 -- CCS contracted to the AMS budget (w=100, d=6) on the
synthetic Fp protocol of Table 2 (D=10^4, N=10^7, 4 distributions), p=3,4,5.
Demonstrates the skewed-data ordering survives equal budgets.

Verbatim copy of the original producer.  PROVENANCE NOTE: the absolute
numbers in results/experiments/exp4_equal_budget.expected.txt were produced
in the paper's original environment and are NOT bit-reproducible here -- the
script is deterministic, but the archived values differ from any current
numpy (verified: the unmodified original producer under numpy 1.26.4 yields
this script's numbers, not the archived ones).  The paper only claims
DIRECTIONAL facts for this experiment, and those hold in this environment
(checked against the AMS reference values in experiments/run_all.py):
contracted CCS stays ahead of AMS on every skewed column and collapses on
uniform at p>=4.  The original run_all.py likewise verified exp4
directionally rather than verbatim.
"""
import numpy as np

D, N = 10_000, 10_000_000
T = 10

def gen(dist, seed=7):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        a = rng.integers(0, D, size=N)
    elif dist.startswith("zipf"):
        wts = np.arange(1, D + 1) ** (-float(dist[4:]))
        a = rng.choice(D, size=N, p=wts / wts.sum())
    elif dist == "mix":
        a1 = rng.integers(0, D, size=N // 2)
        wts = np.arange(1, D + 1) ** (-1.1)
        a2 = rng.choice(D, size=N - N // 2, p=wts / wts.sum())
        a = np.concatenate([a1, a2])
    k, c = np.unique(a, return_counts=True)
    return k, c

def ccs_fp(k, c, p, w, d, rng):
    U = len(k)
    truth = float((c.astype(np.float64) ** p).sum())
    re = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        cr = np.bincount(h, weights=c * np.cos(ang), minlength=w)
        ci = np.bincount(h, weights=c * np.sin(ang), minlength=w)
        pr, pi = 1.0 + 0 * cr, 0.0 * cr
        for _ in range(p):
            pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
        re[j] = pr.sum()
    return abs(float(np.median(re)) - truth) / truth

def main():
    for dist in ("uniform", "zipf1.1", "zipf1.5", "mix"):
        k, c = gen(dist)
        for p in (3, 4, 5):
            rng = np.random.default_rng(500)
            E = np.array([ccs_fp(k, c, p, 100, 6, rng) for _ in range(T)])
            print(f"CCS w=100 d=6 {dist} p={p}: mean={E.mean():.3e} med={np.median(E):.3e}", flush=True)

if __name__ == "__main__":
    main()
