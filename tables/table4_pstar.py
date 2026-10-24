r"""tables/table4_pstar.py -- reproduce paper Table 4 (table:pstar).

Relative error of p-star join size estimation (p=3, w=256, d=64, N=10^7
rows per column); G/A is T-MCCS's error divided by the plain median's.

The published table composes TWO archived producer protocols over the same
bundled Lp columns (both are reproduced here, bit-exact):

  CCS / T-MCCS / G-A columns
      paper_code/python/review/rev_pstar.py  "full 256 64 100"
      (one rng seeded 7919 per combo; T=100 hash families; the per-trial
      Rand20 permutation draw is replicated so the RNG stream stays aligned)
      archived log: results/rev_pstar_full.txt  -- matches the paper exactly.

  Per-edge / Shared-sign / Cubes columns
      paper_code/experiments/exp9_pstar_baselines.py  (per-combo runs)
      (per-edge seeds 8100+t, shared-sign 8200+t, cubes 8300+t, each T=100)
      archived logs: results/exp9_pstar_baselines_synth01.txt, exp9_c2..c5.txt
      -- match the paper exactly.

Usage:
    python3 tables/table4_pstar.py                # full protocol (T=100)
    python3 tables/table4_pstar.py --trials 30    # smoke (numbers drift)
    python3 tables/table4_pstar.py --jobs 6       # parallel over combos
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
from loaders import lp_column, star_truth

P, W, DEPTH = 3, 256, 64
SUF = ["", "2", "3"]
COMBOS = [("uniform", "uniform", "uniform"),
          ("uniform", "uniform", "zipf1.1"),
          ("uniform", "zipf1.1", "zipf1.5"),
          ("uniform", "zipf1.5", "zipf1.5"),
          ("zipf1.1", "zipf1.1", "zipf1.1"),
          ("zipf1.5", "zipf1.5", "zipf1.5")]
_CN = {"uniform": "u", "zipf1.1": "z1.1", "zipf1.5": "z1.5"}

# Published Table 4 values (paper table:pstar): CCS, T-MCCS, G/A, Per-edge, Shared.
PAPER = {
    "uuu":         (1.03e-01, 1.05e-01, 1.02, 2.96e-02, 9.72e-01),
    "uuz1.1":      (6.87e-01, 6.70e-01, 0.98, 3.63e-01, 1.27),
    "uz1.1z1.5":   (3.98e-01, 4.12e-01, 1.03, 4.40e-01, 1.08),
    "uz1.5z1.5":   (5.07e-01, 5.20e-01, 1.03, 4.50e-01, 1.13),
    "z1.1z1.1z1.1": (2.51e-03, 1.54e-03, 0.61, 1.19e-03, 1.09),
    "z1.5z1.5z1.5": (3.13e-04, 1.85e-04, 0.59, 1.17e-04, 1.12),
}


def combo_key(names):
    return "".join(_CN[n] for n in names)


def _load_combo(names):
    """(keys, counts) per column + merged universe + per-column positions.

    Positions index each column's keys into the merged universe, exactly as
    rev_pstar.run_combo does (bincount over the column subset, not the full
    universe -- same values, 3x faster on skewed combos).
    """
    fc = [lp_column(t + ".dat" + s) for t, s in zip(names, SUF)]
    allk = np.unique(np.concatenate([k for k, _ in fc]))
    pos = [np.searchsorted(allk, k) for k, _ in fc]
    return fc, allk, pos


def run_rev_pstar(names, T=100, seed0=7919):
    """Faithful port of rev_pstar.run_combo (CCS/T-MCCS/controls/diagnostics)."""
    fc, allk, pos = _load_combo(names)
    truth = star_truth(fc)
    U = len(allk)
    rng = np.random.default_rng(seed0)
    A = np.empty(T); G = np.empty(T); B = np.empty(T); D = np.empty(T)
    corrs = np.empty(T); pcts = np.empty((T, 5))
    occ_mean, occ_bins = 0.0, np.zeros(5)
    t0 = time.time()
    for t in range(T):
        re = np.empty(DEPTH); im = np.empty(DEPTH)
        for j in range(DEPTH):
            h = rng.integers(0, W, size=U)
            g = rng.integers(0, P, size=U)
            ang = (2.0 * np.pi / P) * g
            gr, gi = np.cos(ang), np.sin(ang)
            pr = np.ones(W); pi = np.zeros(W)
            for (kk, cc), pp in zip(fc, pos):
                cr = np.bincount(h[pp], weights=cc * gr[pp], minlength=W)
                ci = np.bincount(h[pp], weights=cc * gi[pp], minlength=W)
                pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
            re[j] = pr.sum()
            im[j] = abs(pi.sum())
        a = float(np.median(re))
        q1, q3 = np.percentile(im, [25, 75])
        keep = np.nonzero(im <= q3 + 1.5 * (q3 - q1))[0]
        if len(keep) < DEPTH // 2:
            keep = np.argsort(im)[: DEPTH // 2]
        A[t] = abs(a - truth) / truth
        G[t] = abs(float(np.median(re[keep])) - truth) / truth
        B[t] = abs(float(np.median(re[np.argsort(im)[: int(round(DEPTH * 0.8))]]))
                   - truth) / truth
        dk = rng.permutation(DEPTH)[: max(1, int(round(DEPTH * 0.8)))]
        D[t] = abs(float(np.median(re[dk])) - truth) / truth
        corrs[t] = np.corrcoef(im, np.abs(re - truth))[0, 1] if np.std(im) > 0 else 0.0
        pcts[t] = np.percentile(re / truth, [5, 25, 50, 75, 95])
        if t == 0:
            # bucket occupancy of row 0 over column-1 keys (archived diagnostic)
            o = np.bincount(h[pos[0]], minlength=W)
            occ_mean = float(o.mean())
            occ_bins = np.array([(o == 0).mean(), ((o >= 1) & (o <= 2)).mean(),
                                 ((o >= 3) & (o <= 9)).mean(),
                                 ((o >= 10) & (o <= 99)).mean(),
                                 (o >= 100).mean()])
    return dict(name=combo_key(names), truth=truth, U=U,
                A=A.mean(), G=G.mean(), B=B.mean(), D=D.mean(),
                Asd=A.std(ddof=1), Gsd=G.std(ddof=1),
                G_ratio=G.mean() / A.mean(),
                D_ratio=D.mean() / A.mean(),
                win=float(np.mean(G < A)),
                corr=float(np.mean(corrs)), corr_sd=float(np.std(corrs, ddof=1)),
                pcts=np.median(pcts, axis=0).tolist(),
                occ_mean=occ_mean, occ_bins=occ_bins.tolist(),
                sec=time.time() - t0)


def run_baselines(names, T=100):
    """exp9 protocols: per-edge (8100+t), shared-sign (8200+t), cubes (8300+t)."""
    fc, allk, pos = _load_combo(names)
    U = len(allk)
    F = [np.zeros(U) for _ in range(3)]
    for i, (k, c) in enumerate(fc):
        F[i][pos[i]] = c
    truth = star_truth(fc)
    E = np.empty(T); S = np.empty(T); C = np.empty(T)
    NEG_E = NEG_S = NEG_C = 0
    dev = 0.0
    for t in range(T):
        rng = np.random.default_rng(8100 + t)
        est = ccs.peredge_star(F, W, rng, d=DEPTH)
        m = ccs.median(est)
        NEG_E += (m < 0)
        dev += abs(est.mean() - truth) / truth
        E[t] = abs(m - truth) / truth
        rng = np.random.default_rng(8200 + t)
        est = ccs.sharedsign_star(F, W, rng, d=DEPTH)
        m = ccs.median(est)
        NEG_S += (m < 0)
        S[t] = abs(m - truth) / truth
        rng = np.random.default_rng(8300 + t)
        est = ccs.cubes_real(F, W, rng, d=DEPTH, p=P)
        m = ccs.median(est)
        NEG_C += (m < 0)
        C[t] = abs(m - truth) / truth
    return dict(name=combo_key(names), peredge=E.mean(),
                peredge_med=float(np.median(E)), peredge_neg=NEG_E,
                peredge_bias=dev / T,
                sharedsign=S.mean(), sharedsign_med=float(np.median(S)),
                sharedsign_neg=NEG_S,
                cubes=C.mean(), cubes_med=float(np.median(C)), cubes_neg=NEG_C)


def _job(args):
    names, T = args
    r = run_rev_pstar(names, T)
    b = run_baselines(names, T)
    r.update(b)
    return r


def main():
    argv = sys.argv[1:]
    T = 100
    jobs = 1
    if "--trials" in argv:
        T = int(argv[argv.index("--trials") + 1])
    if "--jobs" in argv:
        jobs = int(argv[argv.index("--jobs") + 1])

    print(f"# Table 4 (table:pstar): p={P}, w={W}, d={DEPTH}, T={T} families, "
          f"N=10^7 rows/column", flush=True)
    todo = [(c, T) for c in COMBOS]
    if jobs > 1:
        with Pool(min(jobs, len(todo))) as pool:
            res = pool.map(_job, todo)
    else:
        res = [_job(a) for a in todo]

    # ---- archived-log style table (rev_pstar protocol) ---------------------- #
    print(f"\n# {'combo':>12} {'truth':>11} {'plain':>10} {'T-MCCS':>10} "
          f"{'Fixed20':>10} {'Rand20':>10} {'G/A':>6} {'D/A':>6} {'G<A':>5}")
    for r in res:
        print(f"  {r['name']:>12} {r['truth']:>11.4e} {r['A']:>10.3e} "
              f"{r['G']:>10.3e} {r['B']:>10.3e} {r['D']:>10.3e} "
              f"{r['G_ratio']:>6.3f} {r['D_ratio']:>6.3f} {r['win']:>5.2f}"
              f"   ({r['sec']:.0f}s)", flush=True)

    # ---- exp9 baseline log style -------------------------------------------- #
    print("\n# exp9-style baselines (per-edge / shared-sign / cubes)")
    for r in res:
        print(f"  [{r['name']}] peredge mean={r['peredge']:.4g} "
              f"med={r['peredge_med']:.4g} neg={r['peredge_neg']}/{T} "
              f"bias={r['peredge_bias']:.3g}  sharedsign mean={r['sharedsign']:.4g} "
              f"med={r['sharedsign_med']:.4g} neg={r['sharedsign_neg']}/{T}  "
              f"cubes mean={r['cubes']:.4g} med={r['cubes_med']:.4g} "
              f"neg={r['cubes_neg']}/{T}", flush=True)

    # ---- paper Table 4 facsimile with published values ---------------------- #
    print("\n# paper Table 4 facsimile: reproduced vs published")
    print(f"# {'combo':>12} | {'CCS':>9} {'paper':>9} | {'T-MCCS':>9} {'paper':>9} "
          f"| {'G/A':>5} {'paper':>5} | {'Per-edge':>9} {'paper':>9} "
          f"| {'Shared':>8} {'paper':>8}")
    for r in res:
        pc, pt, pg, pe, ps = PAPER[r["name"]]
        print(f"  {r['name']:>12} | {r['A']:>9.2E} {pc:>9.2E} | {r['G']:>9.2E} "
              f"{pt:>9.2E} | {r['G_ratio']:>5.2f} {pg:>5.2f} | {r['peredge']:>9.2E} "
              f"{pe:>9.2E} | {r['sharedsign']:>8.2f} {ps:>8.2f}", flush=True)

    # ---- diagnostics used by Section 5.4 ------------------------------------ #
    print("\n# diagnostics (rev_pstar protocol)")
    for r in res:
        print(f"  {r['name']:>12} corr(|Im|,err)={r['corr']:+.3f} "
              f"({r['corr_sd']:.2f})  occ_mean={r['occ_mean']:.1f} "
              f"buckets0={r['occ_bins'][0]:.3f} "
              f"Re/truth pct [5,25,50,75,95]={[round(x, 3) for x in r['pcts']]}")
    print("\n# std over trials")
    for r in res:
        print(f"  {r['name']:>12} std(plain)={r['Asd']:.3e} "
              f"std(T-MCCS)={r['Gsd']:.3e}")


if __name__ == "__main__":
    main()
