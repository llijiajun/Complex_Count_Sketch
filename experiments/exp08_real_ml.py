"""Exp 08 -- real MovieLens joins at orders p>=3 (paper Section 5.5, Tables 4-5).

(A) 3-star over three round-robin shards of the bundled ratings (aligned):
    CCS vs byte-matched per-edge real vs shared-sign.
(B) One shared-hash order-3 deployment answering F_3 of every shard, the
    2-join, and the 3-star (paper: "One deployment, three tasks").
(C) 2-joins answered from orders 3,4,5 vs byte-matched Count Sketch
    (Table 5; JoinSketch requires the external C++ and stays in results/).

Usage: python exp08_real_ml.py [families]     (paper: 24; default 12)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import raw
import ccs

P, W, DEPTH = 3, 256, 64
FAMS = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 12

if __name__ == "__main__":
    ru = raw("ML/ratings/userId.gz").astype(np.int64)
    rm = raw("ML/ratings/movieId.gz").astype(np.int64)

    for name, arr in (("userId", ru), ("movieId", rm)):
        kcol = np.unique(arr)
        universe = kcol
        shards = []
        for i in range(3):
            k, c = np.unique(arr[i::3], return_counts=True)
            f = np.zeros(len(universe)); f[np.searchsorted(universe, k)] = c
            shards.append(f)
        truth = float((shards[0] * shards[1] * shards[2]).sum())

        # (A) aligned 3-star
        A, E, S = np.empty(FAMS), np.empty(FAMS), np.empty(FAMS)
        for t in range(FAMS):
            rng = np.random.default_rng(9000 + t)
            re, im = ccs.star_rows(shards, P, W, rng, d=DEPTH)
            A[t] = abs(ccs.median(re) - truth) / truth
            rng = np.random.default_rng(9100 + t)
            est = ccs.peredge_star(shards, W, rng, d=DEPTH)
            E[t] = abs(ccs.median(est) - truth) / truth
            rng = np.random.default_rng(9200 + t)
            est = ccs.sharedsign_star(shards, W, rng, d=DEPTH)
            S[t] = abs(ccs.median(est) - truth) / truth
        print(f"[3star ML {name}] truth={truth:.5g} CCS={A.mean():.4g} "
              f"peredge={E.mean():.4g} sharedsign={S.mean():.4g}", flush=True)

        # (B) one shared-hash deployment, three tasks
        for t in range(1):
            rng = np.random.default_rng(9300 + t)
            fam = ccs.stored_family(shards, P, W, rng, d=DEPTH)
            tasks = ccs.family_tasks(fam, shards, P)
            for i in range(3):
                tf3 = float((shards[i] ** P).sum())
                err = abs(ccs.median(tasks[f"F_p[{i}]"]) - tf3) / tf3
                print(f"[deploy F3 {name} shard{i+1}] truth={tf3:.5g} err={err:.4g}")
            t2 = float((shards[0] * shards[1]).sum())
            err = abs(ccs.median(tasks["join(0,1)"]) - t2) / t2
            print(f"[deploy 2join {name} 0x1] truth={t2:.5g} err={err:.4g}")
            err = abs(ccs.median(tasks["star"]) - truth) / truth
            print(f"[deploy 3star {name}]     truth={truth:.5g} err={err:.4g}",
                  flush=True)

    # (C) 2-joins answered from orders 3,4,5 (Table 5 rows)
    tu = raw("ML/tags/userId.gz").astype(np.int64)
    tm = raw("ML/tags/movieId.gz").astype(np.int64)
    for name, R, S2 in (("userId", ru, tu), ("movieId", rm, tm)):
        universe = np.unique(np.concatenate([np.unique(R), np.unique(S2)]))
        def dense(v):
            k, c = np.unique(v, return_counts=True)
            f = np.zeros(len(universe)); f[np.searchsorted(universe, k)] = c
            return f
        fR, fS = dense(R), dense(S2)
        truth = float((fR * fS).sum())
        line = [f"[2join ML {name}] truth={truth:.5g}"]
        for p, w in ((2, 2048), (3, 1024), (4, 1024), (5, 1024)):
            errs = np.empty(FAMS)
            for t in range(FAMS):
                rng = np.random.default_rng(4000 + 100 * p + t)
                CR, CI = ccs.stored_sketch(fR, p, w, rng, d=DEPTH)
                DS, DI = ccs.stored_sketch(fS, p, w, rng, d=DEPTH)
                rej = ccs.join_rows(CR, CI, DS, DI, p)
                errs[t] = abs(ccs.median(rej) - truth) / truth
            tag = "CS" if p == 2 else f"CCS p={p}"
            line.append(f"{tag} mean={errs.mean():.3e}")
        print("  ".join(line), flush=True)
