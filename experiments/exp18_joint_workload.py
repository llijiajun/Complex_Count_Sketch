r"""experiments/exp18_joint_workload.py -- joint workload: one CCS deployment
serving F_3 (three columns), the 2-join A x B, and the 3-star A x B x C, all
from the SAME three order-3 sketches (w=256, d=64, shared h,g).  Paper
Section 5.5.

Original: paper_code/experiments/exp18_joint_workload.py

Memory accounting: CCS = 3 x 256 x 64 complex counters = 786 KB.
The real toolkit (per-edge Rademacher at byte-matched width) covers the two
join tasks from the same bytes but has NO F_3 solution in the power-sum
family (Theorem 4.2): covering F_3 requires either an exact array over the
key domain (D ~ 2.8M x 4B ~ 11 MB at this workload) or a non-mergeable
AMS-type sampler.  The experiment measures the per-task errors of the single
CCS deployment against exact truths.

Artifact changes vs the original: the Lp columns are read through loaders.py
(bundled gzip files, byte-identical to the originals).  Output matches
results/experiments/exp18_joint_workload.txt.

Usage:
    python3 experiments/exp18_joint_workload.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaders

P, W, DROWS, T = 3, 256, 64, 20

if __name__ == "__main__":
    names = ["zipf1.1", "zipf1.1", "zipf1.1"]
    sufs = ["", "2", "3"]
    fc = [loaders.lp_column(t + ".dat" + s) for t, s in zip(names, sufs)]
    allk = np.unique(np.concatenate([k for k, _ in fc]))
    U = len(allk)
    F = [np.zeros(U) for _ in range(3)]
    for i, (k, c) in enumerate(fc):
        F[i][np.searchsorted(allk, k)] = c
    tF = [float((f ** P).sum()) for f in F]
    t2 = float((F[0] * F[1]).sum())
    t3 = float((F[0] * F[1] * F[2]).sum())
    print(f"keys={U}  F3 errors of tasks: F3(A),F3(B),F3(C),J2(A,B),J3(A,B,C)")
    e = {kk: [] for kk in ("F3A", "F3B", "F3C", "J2", "J3")}
    for t in range(T):
        rng = np.random.default_rng(6000 + t)
        h = rng.integers(0, W, size=U)
        g = rng.integers(0, P, size=U)
        ang = (2.0 * np.pi / P) * g
        gr, gi = np.cos(ang), np.sin(ang)
        Cs = []
        for f in F:
            cr = np.bincount(h, weights=f * gr, minlength=W)
            ci = np.bincount(h, weights=f * gi, minlength=W)
            Cs.append(cr + 1j * ci)
        X = [ (C ** P) for C in Cs ]
        e["F3A"].append(abs(float(np.median(X[0].real.sum())) - tF[0]) / tF[0])
        e["F3B"].append(abs(float(np.median(X[1].real.sum())) - tF[1]) / tF[1])
        e["F3C"].append(abs(float(np.median(X[2].real.sum())) - tF[2]) / tF[2])
        pr = Cs[0] * np.conj(Cs[1])
        e["J2"].append(abs(float(np.median(pr.real.sum())) - t2) / t2)
        pr3 = Cs[0] * Cs[1] * Cs[2]
        e["J3"].append(abs(float(np.median(pr3.real.sum())) - t3) / t3)
    for kk, v in e.items():
        v = np.array(v)
        print(f"{kk}: truth={( {**{'F3A':tF[0],'F3B':tF[1],'F3C':tF[2],'J2':t2,'J3':t3}} )[kk]:.5g}  mean={v.mean():.4g} med={np.median(v):.4g}")
    mb = 3 * W * DROWS * 16 / 1e6
    print(f"memory: 3 sketches = {mb:.2f} MB (complex double); exact F3 array over D={U} keys = {U*4/1e6:.1f} MB")
    print("EXP18_DONE")
