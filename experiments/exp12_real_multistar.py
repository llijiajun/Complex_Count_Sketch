r"""experiments/exp12_real_multistar.py -- real 3- and 5-table star joins at
equal bytes vs the per-edge real sketch (paper Section 5.5).

Original: paper_code/experiments/exp12_real_multistar.py

(A) MovieLens ratings split into FIVE disjoint round-robin shards (aligned
    5-star, p=5), keys userId and movieId.
(B) Kasandr top-FIVE offer categories as verticals (misaligned 5-star, p=5).

Estimators, same protocol as exp9 (p=5 here, w=256 complex d=64, T=24
families): CCS shared-root; per-edge real Rademacher at byte-matched width
2w=512 (4 edges of the 5-node star: column 1 carries the product of its four
incident edge signs, column i carries s1i); shared-sign real product
(biased, E[g^5]=E[g]=0 over {±1}).

Artifact changes vs the original: data loading through loaders.py and the
estimators through ccs.py (same protocols and seeds).  Output matches
results/experiments/exp12_real_multistar.txt.

Usage:
    python3 experiments/exp12_real_multistar.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
import loaders

P = 5
W, D_ROWS, T = 256, 64, 24


def report(tag, truth, F):
    out = [f"[{tag}] truth={truth:.6g}"]
    A = np.empty(T); E = np.empty(T); S = np.empty(T); NEG = 0
    for t in range(T):
        rng = np.random.default_rng(9500 + t)
        re, _ = ccs.star_rows(F, P, W, rng, d=D_ROWS)
        A[t] = abs(ccs.median(re) - truth) / truth
        rng = np.random.default_rng(9600 + t)
        est = ccs.peredge_star(F, W, rng, d=D_ROWS)
        E[t] = abs(ccs.median(est) - truth) / truth
        rng = np.random.default_rng(9700 + t)
        est = ccs.sharedsign_star(F, W, rng, d=D_ROWS)
        m = float(np.median(est))
        NEG += (m < 0)
        S[t] = abs(m - truth) / truth
    out.append(f"CCS mean={A.mean():.4g} peredge mean={E.mean():.4g} med={np.median(E):.4g} "
               f"sharedsign mean={S.mean():.4g} neg={NEG}/{T}")
    print("  ".join(out), flush=True)


def reindex(k, c, kcol):
    f = np.zeros(len(kcol), dtype=np.float64)
    f[np.searchsorted(kcol, k)] = c
    return f


def ml_5star():
    for key in ("userId", "movieId"):
        arr = loaders.raw(os.path.join("ML", "ratings", key + ".gz"))
        kcol = np.unique(arr)
        shards = [np.unique(arr[i::5], return_counts=True) for i in range(5)]
        F = [reindex(k, c, kcol) for k, c in shards]
        truth = float(np.prod(np.vstack([f + 1e-300 for f in F]), axis=0).sum())
        report(f"5star ML {key}", truth, F)


def kas_5cat():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import exp2_kasandr_3star as K
    users, cats, times, U = K.load()
    cc = np.bincount(cats)
    top = np.argsort(cc)[::-1][:5]
    Fs = []
    for cv in top:
        k, c = np.unique(users[cats == cv], return_counts=True)
        f = np.zeros(U); f[k] = c
        Fs.append(f)
    truth = float(np.prod(np.vstack([f + 1e-300 for f in Fs]), axis=0).sum())
    report("5star KAS cat", truth, Fs)


if __name__ == "__main__":
    ml_5star()
    kas_5cat()
    print("EXP12_DONE", flush=True)
