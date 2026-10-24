r"""experiments/exp2_kasandr_3star.py -- real 3-star joins on Kasandr.

Original: paper_code/experiments/exp2_kasandr_3star.py
(train_de.csv, 15.84M rows, key=userid).

(a) 3 disjoint TIME-WINDOW shards (aligned deployment case: near-identical
    supports);
(b) 3 top-CATEGORY verticals (different user populations -> misaligned
    supports).
Reports per-fragment stats, support overlaps, exact truth J (full scan),
CCS p=3 plain vs T-MCCS (w=256, d=64, T=24 hash families).

Artifact changes vs the original: the CSV is pre-extracted into
data/kasandr_essential.npz (user/category ids by first appearance, times as
YYYYMMDD integers -- exactly what the original load() computed); the CCS row
estimator and T-MCCS truncation now come from ccs.py (bit-identical
protocol).  Output matches results/experiments/exp2_kasandr_3star.txt.

Usage:
    python3 experiments/exp2_kasandr_3star.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs
import loaders


def load():
    """(users, cats, times, n_users): first-appearance id encoding."""
    return loaders.kasandr()


def frag_stats(mask, users, U, label):
    k, c = np.unique(users[mask], return_counts=True)
    f = np.zeros(U, dtype=np.int64); f[k] = c
    print(f"  {label}: rows={int(mask.sum())} distinct-users={len(k)} peak-f={int(c.max())}", flush=True)
    return f


def run_construct(name, F, T=24, w=256, d=64):
    J = float((F[0] * F[1] * F[2]).sum())
    A = np.empty(T); G = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(9000 + t)
        re, im = ccs.star_rows(F, 3, w, rng, d=d)
        A[t] = abs(ccs.median(re) - J) / J
        G[t] = abs(ccs.tmccs(re, im) - J) / J
    print(f"{name}: truth J={J:.6g}  CCS plain: mean={A.mean():.4f} med={np.median(A):.4f}  "
          f"T-MCCS: mean={G.mean():.4f} med={np.median(G):.4f}  T<plain win={float(np.mean(G<A)):.2f}", flush=True)


if __name__ == "__main__":
    print("loading Kasandr...", flush=True)
    t0 = time.time()
    users, cats, times, U = load()
    print(f"loaded {len(users)} rows, {U} users, {len(set(cats.tolist()))} categories, {time.time()-t0:.0f}s", flush=True)

    q1t, q2t = np.percentile(times, [100/3, 200/3])
    masks_t = [times <= q1t, (times > q1t) & (times <= q2t), times > q2t]
    Ft = [frag_stats(m, users, U, f"time-window {i+1}") for i, m in enumerate(masks_t)]
    S = [set(np.nonzero(f)[0].tolist()) for f in Ft]
    print(f"  support overlaps |S1∩S2|={len(S[0]&S[1])} |S1∩S3|={len(S[0]&S[2])} |S2∩S3|={len(S[1]&S[2])}")
    run_construct("TIME-WINDOW shards (aligned deployment case)", Ft)

    cc = np.bincount(cats)
    top = np.argsort(cc)[::-1][:3]
    print("top-3 categories:", top.tolist(), "sizes:", cc[top].tolist(), flush=True)
    masks_c = [cats == c for c in top]
    Fc = [frag_stats(m, users, U, f"category {c}") for c, m in zip(top, masks_c)]
    S = [set(np.nonzero(f)[0].tolist()) for f in Fc]
    print(f"  support overlaps |S1∩S2|={len(S[0]&S[1])} |S1∩S3|={len(S[0]&S[2])} |S2∩S3|={len(S[1]&S[2])}")
    run_construct("CATEGORY verticals (misaligned)", Fc)
    print("ALL_DONE", flush=True)
