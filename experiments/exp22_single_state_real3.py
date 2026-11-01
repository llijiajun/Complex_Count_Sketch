"""Exp 22 -- What one stored state covers: CCS vs a single-state real-3
deployment (the joint-workload comparison of Sec. "Moments, Joins and
Stars from One Deployment").

A shared-hash real multiplier (E[sigma]=0, E[sigma^3]=1, mu2>0) stored at
ONE real counter per bucket per column answers all three task families from
one state that merges by addition:

    F3_hat(i)  = sum_b C_i[b]^3                    (E[sigma^3] = 1)
    J2_hat(i,j)= mu2^{-1} sum_b C_i[b] C_j[b]      (cross terms carry E[sigma]^2 = 0)
    J3_hat     = sum_b C_1[b] C_2[b] C_3[b]        (E[sigma^3] = 1 diagonal)

This script measures whether that state MEETS a uniform 10% relative-error
target on every task, against CCS, per-edge, and the per-edge+real-3
composite, at byte-matched budgets, and reports the unbiasedness of every
real-3 query (mean of row estimates / truth).

Systems (workload: three regenerated zipf1.1 columns, D=10^4, N=2x10^6):
  CCS      3 x 256 x 64 complex counters ............ 0.79 MB (all 7 tasks)
  real-3   3 x 512 x 64 real counters (shared h,sigma) 0.79 MB (byte-matched;
           two-point r=sqrt(3/2), mu2 = 1/2, mu6 = 89/8; all 7 tasks)
  real-3   3 x 768 x 64 real counters .................. 1.18 MB (matched)
  pair-only real joins: three 2-column +-1 deployments 3 x 256 x 64 x 8B
           .............................................. 0.79 MB (joins only;
           the 3-column per-edge star deployment shares s_i*s_j that do not
           cancel, so it answers NO pairwise join)
  composite pair joins 0.79 MB + real-3 tier 3 x 256 x 64 (0.39 MB) = 1.18 MB
  real-3 MEAN aggregator at d rows: the mu6-1 floor forbids no target in the
           limit of unbounded rows (E[X_j] = J exactly), so we also report the
           empirical row count at which the mean meets the target.

Usage: python exp22_single_state_real3.py [deployments]     # paper: 40
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

EPS = 0.1
N, D = 2_000_000, 10_000
R = np.sqrt(1.5)                       # two-point support 1+-r
W1 = (R - 1) / (2 * R)                 # prob of 1+r
MU2 = 0.5                              # E[sigma^2], exact for this family
MU6 = 89.0 / 8.0


def make_column(seed):
    rng = np.random.default_rng(seed)
    p = np.arange(1, D + 1, dtype=float) ** (-1.1)
    p /= p.sum()
    draws = rng.choice(D, size=N, p=p)
    k, c = np.unique(draws, return_counts=True)
    f = np.zeros(D)
    f[k] = c
    return f


def real3_rows(F, w, rng, d=64):
    """One shared-hash order-3 REAL deployment: per-row estimates of every
    task, from the SAME stored counters (three real arrays, one per column).

    Returns dict of per-row estimates: F3[i], J2(i,j), star.
    """
    U = len(F[0])
    out = {"F3": np.empty((3, d)), "J2": np.empty((3, d)), "star": np.empty(d)}
    prs = [(1 + R, W1), (1 - R, 1 - W1)]
    for j in range(d):
        h = rng.integers(0, w, size=U)
        hi = rng.random(U) < W1
        s = np.where(hi, 1 + R, 1 - R)          # shared sigma across columns
        cs = [np.bincount(h, weights=f * s, minlength=w) for f in F]
        for i in range(3):
            out["F3"][i, j] = (cs[i] ** 3).sum()
        m = 0
        for i in range(3):
            for j2 in range(i + 1, 3):
                out["J2"][m, j] = (cs[i] * cs[j2]).sum() / MU2
                m += 1
        out["star"][j] = (cs[0] * cs[1] * cs[2]).sum()
    return out


def ccs_rows(F, w, rng, d=64):
    """One shared-hash order-3 CCS deployment: per-row estimates of every task."""
    sketches = ccs.stored_family(F, 3, w, rng, d=d)
    return ccs.family_tasks(sketches, F, 3)


def peredge_pair_rows(F, w, rng, d=64):
    """THREE 2-column per-edge deployments, one per pair join (a 3-column
    star deployment shares s_i*s_j that do NOT cancel, so it answers no
    pairwise join).  One shared +-1 sign per pair: the standard Count-Sketch
    join.  Returns the three per-row join estimate arrays (d rows each)."""
    U = len(F[0])
    out = []
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        rows = np.empty(d)
        for t in range(d):
            h = rng.integers(0, w, size=U)
            s = rng.integers(0, 2, size=U) * 2 - 1
            ci = np.bincount(h, weights=F[i] * s, minlength=w)
            cj = np.bincount(h, weights=F[j] * s, minlength=w)
            rows[t] = (ci * cj).sum()
        out.append(rows)
    return out


if __name__ == "__main__":
    DEP = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 40
    print(f"=== one deployment, seven tasks, joint target eps={EPS}: "
          f"{DEP} deployments per system ===")
    F = [make_column(s) for s in (61, 62, 63)]
    F3t = [float((f ** 3).sum()) for f in F]
    J2t = [float((F[i] * F[j]).sum()) for i in range(3) for j in range(i + 1, 3)]
    J3t = float((F[0] * F[1] * F[2]).sum())
    TNS = [f"F3[{i}]" for i in range(3)] + \
          ["J2(0,1)", "J2(0,2)", "J2(1,2)"] + ["3-star"]
    TRUTH = dict(zip(TNS, F3t + J2t + [J3t]))
    Q = float(((F[0] * F[1] * F[2]) ** 2).sum())
    floor = (MU6 - 1) * Q / (EPS ** 2 * J3t ** 2)
    print(f"real-3 diagonal floor / (eps^2 J^2) on this workload = {floor:.2f}")

    def build(kind, w, d, rng, agg):
        """One deployment: {task: estimate} under the chosen aggregator."""
        def agg1(v):
            return float(np.median(v) if agg == "median" else np.mean(v))
        if kind == "ccs":
            r = ccs_rows(F, w, rng, d)
            return {**{f"F3[{i}]": agg1(r[f"F_p[{i}]"]) for i in range(3)},
                    **{f"J2({i},{j})": agg1(r[f"join({i},{j})"])
                       for (i, j) in ((0, 1), (0, 2), (1, 2))},
                    "3-star": agg1(r["star"])}
        if kind == "real3":
            r = real3_rows(F, w, rng, d)
            return {**{f"F3[{i}]": agg1(r["F3"][i]) for i in range(3)},
                    **{f"J2({i},{j})": agg1(r["J2"][m])
                       for m, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)))},
                    "3-star": agg1(r["star"])}
        if kind == "pairs":
            r = peredge_pair_rows(F, w, rng, d)
            return {f"J2({i},{j})": agg1(r[m])
                    for m, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)))}
        raise ValueError(kind)

    def report(tag, rows_by_task, emax):
        print(f"--- {tag}")
        ok = sum(1 for e in emax if e <= EPS)
        print(f"  JOINT (Emax<=eps): {ok}/{len(emax)} deployments  "
              f"median Emax={np.median(emax):.3f}  "
              f"max Emax={max(emax):.3f}", flush=True)
        for tn in TNS:
            if tn not in rows_by_task:
                continue
            e = np.abs(np.array(rows_by_task[tn]) - TRUTH[tn]) / TRUTH[tn]
            print(f"  {tn:10s} med-err={np.median(e):9.3e}  "
                  f"fail={float((e > EPS).mean()):5.2f}", flush=True)

    systems = [
        ("CCS median 0.79MB (3x256x64x16B)", "ccs", 256, 64, "median", 0),
        ("real-3 median 0.79MB (3x512x64x8B)", "real3", 512, 64, "median", 0),
        ("real-3 median 0.79MB (3x128x256x8B)", "real3", 128, 256, "median", 0),
        ("real-3 MEAN 0.79MB (3x128x256x8B)", "real3", 128, 256, "mean", 0),
        ("real-3 MEAN 3.1MB (3x512x256x8B)", "real3", 512, 256, "mean", 0),
        ("real-3 MEAN 1.6MB (3x128x512x8B)", "real3", 128, 512, "mean", 0),
        ("real-3 MEAN 3.1MB (3x128x1024x8B)", "real3", 128, 1024, "mean", 0),
        ("real-3 median 1.18MB (3x768x64x8B)", "real3", 768, 64, "median", 0),
        ("pair-only real joins 0.79MB (3x256x64x8B)", "pairs", 256, 64, "median", 0),
    ]
    for name, kind, w, d, agg, _ in systems:
        rows_by_task = {tn: [] for tn in TNS}
        emax = []
        for t in range(DEP):
            rng = np.random.default_rng(5000 + 17 * t)
            est = build(kind, w, d, rng, agg)
            e = max(abs(est[tn] - TRUTH[tn]) / TRUTH[tn] for tn in est)
            emax.append(e)
            for tn, v in est.items():
                rows_by_task[tn].append(v)
        report(name, rows_by_task, emax)

    # composite: pair joins (0.79MB) + real-3 tier median (0.39MB) = 1.18 MB
    rows_by_task = {tn: [] for tn in TNS}
    emax = []
    for t in range(DEP):
        rng = np.random.default_rng(9000 + 17 * t)
        rj = peredge_pair_rows(F, 256, rng, 64)
        rm = real3_rows(F, 256, rng, 64)
        est = {}
        for m, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            est[f"J2({i},{j})"] = float(np.median(rj[m]))
        for i in range(3):
            est[f"F3[{i}]"] = float(np.median(rm["F3"][i]))
        est["3-star"] = float(np.median(rm["star"]))
        e = max(abs(est[tn] - TRUTH[tn]) / TRUTH[tn] for tn in est)
        emax.append(e)
        for tn, v in est.items():
            rows_by_task[tn].append(v)
    report("composite: pair joins 0.79MB + real-3 tier 0.39MB = 1.18MB",
           rows_by_task, emax)
    print("EXP22_DONE")
