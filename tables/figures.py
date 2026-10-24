r"""tables/figures.py -- reproduce the two paper figures.

fig_sweep.pdf : width sweep of the CCS F_p error (exp8 part b: zipf1.1,
                D=10^4, N=10^7, d=64, mean of 10 runs) with reference
                slopes w^-1/2 and w^-1.
fig_regime.pdf: T-MCCS / plain-median error ratio vs Zipf alpha (exp3
                protocol: D=10^3, N=10^5, w=256, d=64, 1000 runs),
                reference line at 1.0.

Both figures were cut from the final 12-page version of VLDB.tex (the PDFs
still ship in papers/figs/).  This script regenerates them.  By default it
plots the values parsed from the bundled archived logs, i.e. exactly the
numbers the original make_figs.py hardcoded:

    results/figures/exp8_fp_ci_sweep.txt          (fig_sweep, part b)
    results/table3/exp3_tmccs_stats.expected.txt  (fig_regime)

With --recompute it re-derives both datasets from scratch (same generators,
same seeds, same RNG draw order) and checks them against the archives at the
archived precision before plotting.

Usage:
    python3 tables/figures.py                  # plot from archives (fast)
    python3 tables/figures.py --recompute      # re-run + verify (~4 min)
    python3 tables/figures.py --out figs_out   # choose output directory
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP8_LOG = os.path.join(ROOT, "results", "figures", "exp8_fp_ci_sweep.txt")
EXP3_LOG = os.path.join(ROOT, "results", "table3", "exp3_tmccs_stats.expected.txt")

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})

# exp8 part (b) protocol: zipf1.1, D=10^4, N=10^7, d=64, 10 runs per width.
SWEEP_WS = (128, 256, 512, 1024, 2048)
SWEEP_D, SWEEP_REPS = 64, 10
# exp3 protocol: D=10^3, N=10^5, w=256, d=64, 1000 runs per (p, alpha).
REGIME_ALPHAS = (0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0)
REGIME_D, REGIME_N, REGIME_W, REGIME_DEPTH = 1000, 100_000, 256, 64
REGIME_TRIALS = 1000


# --------------------------------------------------------------------------- #
# archived values (what the original make_figs.py hardcoded)                  #
# --------------------------------------------------------------------------- #
def sweep_from_archive():
    """{p: [err per w in SWEEP_WS]} parsed from results/figures/exp8 log."""
    out = {}
    with open(EXP8_LOG) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("p="):
                continue
            head, rest = line.split(":", 1)
            p = int(head[2:])
            vals = []
            for tok in rest.split():
                vals.append(float(tok.split(":")[1]))
            out[p] = vals
    assert sorted(out) == [3, 4, 5], out
    for p, v in out.items():
        assert len(v) == len(SWEEP_WS), (p, v)
    return out


def regime_from_archive():
    """{p: (ccs_err[7], tmccs_err[7])} parsed from the exp3 expected log."""
    ccs_e, tm_e = {}, {}
    with open(EXP3_LOG) as fh:
        p = None
        for line in fh:
            line = line.strip()
            if line.startswith("--- p="):
                p = int(line[6:-4])
                ccs_e[p], tm_e[p] = [], []
            elif line.startswith("a="):
                # a=0.0: plain=3.7526e-02 tmccs=3.8432e-02 ratio_of_means=...
                plain = float(line.split("plain=")[1].split()[0])
                tm = float(line.split("tmccs=")[1].split()[0])
                ccs_e[p].append(plain)
                tm_e[p].append(tm)
    for p in (3, 4, 5):
        assert len(ccs_e[p]) == len(REGIME_ALPHAS), (p, ccs_e[p])
        assert len(tm_e[p]) == len(REGIME_ALPHAS), (p, tm_e[p])
    return {p: (np.array(ccs_e[p]), np.array(tm_e[p])) for p in (3, 4, 5)}


# --------------------------------------------------------------------------- #
# recomputation (identical protocols, from scratch)                           #
# --------------------------------------------------------------------------- #
def sweep_recompute():
    """exp8 part (b): per-row interleaved F_p estimator, one rng per run.

    exp8's gen() returns (distinct keys, counts); the estimator draws hashes
    over the distinct keys only, so the frequency vector must be compacted
    (zero-frequency keys would otherwise consume RNG draws).
    """
    freq = ccs.zipf_freq(10_000, 10_000_000, 1.1, seed=7)   # gen('zipf1.1')
    freq = freq[np.nonzero(freq)[0]]
    out = {}
    for p in (3, 4, 5):
        truth = float((freq ** p).sum())
        vals = []
        for w in SWEEP_WS:
            errs = np.empty(SWEEP_REPS)
            for t in range(SWEEP_REPS):
                re, _ = ccs.fp_rows_interleaved(freq, p, w,
                                                np.random.default_rng(8000 + t),
                                                d=SWEEP_D)
                errs[t] = abs(ccs.median(re) - truth) / truth
            vals.append(errs.mean())
        out[p] = np.array(vals)
    return out


def regime_recompute(trials=REGIME_TRIALS):
    """exp3 protocol (plain / T-MCCS columns of Table 3)."""
    out = {}
    for p in (3, 4, 5):
        ccs_e, tm_e = [], []
        for alpha in REGIME_ALPHAS:
            freq = ccs.zipf_freq(REGIME_D, REGIME_N, alpha, seed=12345)
            truth = float((freq ** p).sum())
            A = np.empty(trials)
            G = np.empty(trials)
            for t in range(trials):
                re, im = ccs.ccs_rows_batch(freq, p, REGIME_W, REGIME_DEPTH,
                                            1000 + t)
                A[t] = abs(ccs.median(re) - truth) / truth
                G[t] = abs(ccs.tmccs(re, im) - truth) / truth
            ccs_e.append(A.mean())
            tm_e.append(G.mean())
        out[p] = (np.array(ccs_e), np.array(tm_e))
    return out


def verify(rtol=2e-3):
    """Recompute both datasets and compare with the bundled archives.

    The archived logs carry 4 significant digits and make_figs.py rounded
    them to 3; we compare at 3 significant digits (rtol 2e-3), which is the
    precision the paper figures themselves were drawn at.
    """
    print("# verifying recomputed figure data against bundled archives "
          "(3 significant digits)")
    ok = True
    sw_a, sw_r = sweep_from_archive(), sweep_recompute()
    for p in (3, 4, 5):
        for j, w in enumerate(SWEEP_WS):
            got, want = sw_r[p][j], sw_a[p][j]
            good = abs(got - want) <= rtol * abs(want)
            ok &= good
            print(f"  sweep p={p} w={w}: recomputed {got:.3e} archived "
                  f"{want:.3e} {'OK' if good else 'MISMATCH'}")
    rg_a, rg_r = regime_from_archive(), regime_recompute()
    for p in (3, 4, 5):
        for j, alpha in enumerate(REGIME_ALPHAS):
            for k, nm in ((0, "plain"), (1, "T-MCCS")):
                got, want = rg_r[p][k][j], rg_a[p][k][j]
                good = abs(got - want) <= rtol * abs(want)
                ok &= good
                print(f"  regime p={p} a={alpha} {nm}: recomputed {got:.3e} "
                      f"archived {want:.3e} {'OK' if good else 'MISMATCH'}")
    print(f"# verify: {'ALL OK' if ok else 'FAILURES PRESENT'}")
    return ok


# --------------------------------------------------------------------------- #
# plotting (layout of papers/figs/*.pdf)                                      #
# --------------------------------------------------------------------------- #
def fig_sweep(err, out):
    w = np.array(SWEEP_WS)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    marks = {3: "o", 4: "s", 5: "^"}
    for p, e in err.items():
        ax.loglog(w, e, marks[p] + "-", ms=3.2, lw=0.9, label=f"$p={p}$")
    ax.loglog(w, err[3][0] * (w / w[0]) ** (-0.5), ":", lw=0.8, color="gray")
    ax.loglog(w, err[3][0] * (w / w[0]) ** (-1.0), "--", lw=0.8, color="gray")
    ax.text(1500, err[3][0] * (1500 / 128) ** (-0.5) * 1.35, r"$w^{-1/2}$",
            fontsize=7, color="gray")
    ax.text(1500, err[3][0] * (1500 / 128) ** (-1.0) * 0.75, r"$w^{-1}$",
            fontsize=7, color="gray")
    ax.set_xlabel("width $w$ (counters per row)")
    ax.set_ylabel("mean relative error")
    ax.set_title("zipf1.1, $d=64$, 10 runs", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.tick_params(direction="in")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig_sweep.pdf"))
    fig.savefig(os.path.join(out, "fig_sweep.png"), dpi=200)
    plt.close(fig)


def fig_regime(data, out):
    alpha = np.array(REGIME_ALPHAS)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    marks = {3: "o", 4: "s", 5: "^"}
    for p in (3, 4, 5):
        ccs_e, tm_e = data[p]
        ax.plot(alpha, tm_e / ccs_e, marks[p] + "-", ms=3.2, lw=0.9,
                label=f"$p={p}$")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.axvspan(0.8, 2.05, color="0.92", lw=0)
    ax.text(1.28, 0.55, "signal regime", fontsize=7, color="0.35",
            ha="center")
    ax.text(0.28, 1.055, "safe-guard regime", fontsize=7, color="0.35",
            ha="center")
    ax.set_xlabel(r"Zipf skewness $\alpha$")
    ax.set_ylabel("T-MCCS / plain median")
    ax.set_ylim(0.4, 1.12)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(direction="in")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig_regime.pdf"))
    fig.savefig(os.path.join(out, "fig_regime.png"), dpi=200)
    plt.close(fig)


def main():
    argv = sys.argv[1:]
    out = os.path.join(HERE, "figs_out")
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)

    if "--recompute" in argv:
        if not verify():
            print("# recomputation disagrees with the archives -- aborting",
                  file=sys.stderr)
            sys.exit(1)
        sweep = sweep_recompute()
        regime = regime_recompute()
    else:
        sweep = sweep_from_archive()
        regime = regime_from_archive()
        print("# plotting values parsed from the bundled archived logs "
              "(pass --recompute to re-derive and verify them)")

    fig_sweep(sweep, out)
    fig_regime(regime, out)
    print(f"# figures written to {out}")


if __name__ == "__main__":
    main()
