"""Exp 20 -- Sizing an F_p deployment from the exact variance
(paper Table 3, "Sizing at eps=0.1, delta=0.01").

Protocol (paper Sec. 5.3 "Sizing a Deployment from the Exact Variance"):
  * instances : uniform and zipf1.1 over D=10^4 keys, N=2x10^6 draws (seeded)
  * rule      : Var[Re X_j] = A/w + B/w^2 with A = (9/2) S2, B = 18 S3 and
                S2 = F2 F4 + F3^2 - 2 F6,  S3 = (F2^3 - 3 F2 F4 + 2 F6)/6
                (Eq. eq:AB-moments); choose the least w with
                q(w) = (A/w + B/w^2) / (eps^2 F3^2) <= 1/4 and
                d* = ceil(8 ln(1/delta)) = 37 rows.
  * configs   : prior rule w = D^{1-2/p}/eps^2 (d=6); fixed default 256x64;
                ideal sizing from the TRUE moments; pilot sizing where the
                four moments come from order-2/3/4/6 sketches of
                w=512, d=4 (32 KB each, 128 KB charged, matching the KB
                column of Table 3: deployment KB + 128).
  * metrics   : ratio = measured / predicted per-row variance, one pooled
                variance over all rows of a batch, median over 5 independent
                seed batches (bracket = batch min/max).  On the skewed column
                the row variance is itself heavy-tailed, so batch ratios
                scatter widely in either direction -- the paper quotes this
                imprecision explicitly (a Monte-Carlo estimate of a
                heavy-tailed variance is itself imprecise); the sizing
                verdicts below do not depend on it.
                fail = fraction of the deployments with relative error
                > eps; p95 err = 95th percentile of the relative errors.

Usage: python exp20_fp_sizing.py [deployments_per_batch]   # paper: 5 batches x 30
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

EPS, DELTA = 0.1, 0.01
DSTAR = int(np.ceil(8 * np.log(1 / DELTA)))            # 37
N, D = 2_000_000, 10_000
PILOT_W, PILOT_D, PILOT_ORDERS = 512, 4, (2, 3, 4, 6)  # 32 KB per order
PILOT_KB = len(PILOT_ORDERS) * PILOT_W * PILOT_D * 16 / 1024   # 128 KB


def make_column(name, seed):
    """Zipf(alpha) or uniform over D keys, N draws, seeded."""
    rng = np.random.default_rng(seed)
    if name == "uniform":
        draws = rng.integers(0, D, size=N)
    else:
        alpha = float(name[4:])
        p = np.arange(1, D + 1, dtype=float) ** (-alpha)
        p /= p.sum()
        draws = rng.choice(D, size=N, p=p)
    k, c = np.unique(draws, return_counts=True)
    f = np.zeros(D)
    f[k] = c
    return f


def ab_from_moments(F2, F3, F4, F6):
    S2 = F2 * F4 + F3 ** 2 - 2 * F6
    S3 = (F2 ** 3 - 3 * F2 * F4 + 2 * F6) / 6
    return 4.5 * S2, 18.0 * S3


def width_rule(A, B, F3):
    """Least w with (A/w + B/w^2) <= eps^2 F3^2 / 4 (closed form, ceiled)."""
    T = EPS ** 2 * F3 ** 2 / 4
    return int(np.ceil((A + np.sqrt(A * A + 4 * B * T)) / (2 * T)))


def pilot_moments(f):
    """Recover F2, F3, F4, F6 from pilot CCS sketches (128 KB charged)."""
    out = []
    for p in PILOT_ORDERS:
        rng = np.random.default_rng(90_000 + p)
        re, _ = ccs.f_p_rows(f, p, PILOT_W, rng, d=PILOT_D)
        out.append(float(np.median(re)))
    return out


def evaluate(f, A, B, w, d, truth, per_batch=30, seed0=1000):
    """per_batch x 5 deployments; returns fail, p95, and the 5 batch ratios.

    Each batch ratio pools the variance over all per_batch*d rows of the
    batch (one number per batch); the paper reports their median with the
    bracket as spread.
    """
    pred = A / w + B / w ** 2
    errs, ratios = [], []
    for b in range(5):
        rows = np.empty(per_batch * d)
        for t in range(per_batch):
            rng = np.random.default_rng(seed0 + 97 * b + t)
            re, _ = ccs.f_p_rows(f, 3, w, rng, d=d)
            errs.append(abs(float(np.median(re)) - truth) / truth)
            rows[t * d:(t + 1) * d] = re
        ratios.append(float(np.var(rows, ddof=1)) / pred)
    errs = np.array(errs)
    return float((errs > EPS).mean()), float(np.percentile(errs, 95)), ratios


if __name__ == "__main__":
    # deployments PER BATCH; the paper runs 5 batches x 30 = 150 per row
    PER = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 30
    print(f"=== F_p sizing (Table 3): eps={EPS}, delta={DELTA}, d*={DSTAR}, "
          f"{5 * PER} deployments per row ===")
    header = (f"{'column':9s} {'configuration':30s} {'w':>7s} {'d':>3s} "
              f"{'KB':>7s} {'ratio':>17s} {'fail':>6s} {'p95 err':>10s}")
    print(header)
    for name in ("zipf1.1", "uniform"):
        f = make_column(name, seed=42)
        F2, F3, F4, F6 = (float((f ** k).sum()) for k in (2, 3, 4, 6))
        truth = F3
        A, B = ab_from_moments(F2, F3, F4, F6)
        w_prior = int(np.ceil(D ** (1 - 2 / 3) / EPS ** 2))
        w_ideal = width_rule(A, B, F3)
        F2p, F3p, F4p, F6p = pilot_moments(f)
        Ap, Bp = ab_from_moments(F2p, F3p, F4p, F6p)
        w_pilot = width_rule(Ap, Bp, F3p)
        rows = [
            ("prior rule D^{1-2/p}/eps^2", w_prior, 6, 0.0),
            ("fixed default", 256, 64, 0.0),
            ("ideal sizing (true moments)", w_ideal, DSTAR, 0.0),
            ("pilot sizing (+pilot bytes)", w_pilot, DSTAR, PILOT_KB),
        ]
        for cfg, w, d, extra_kb in rows:
            A_use, B_use = (A, B) if "pilot" not in cfg else (Ap, Bp)
            fail, p95, ratios = evaluate(f, A_use, B_use, w, d, truth, per_batch=PER)
            kb = w * d * 16 / 1024 + extra_kb
            print(f"{name:9s} {cfg:30s} {w:7d} {d:3d} {kb:7.0f} "
                  f"{np.median(ratios):7.2f}[{min(ratios):.2f},{max(ratios):.2f}] "
                  f"{fail:6.3f} {p95:10.2e}", flush=True)
    print("EXP20_DONE")
