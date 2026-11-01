"""Exp 13 -- joint workload: one CCS deployment serving F_3 (three columns),
the 2-join, and the 3-star from the SAME three stored order-3 sketches
(paper Section 5.5, "One deployment, three tasks").
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import lp_column, merge_universe
import ccs

P, W, DEPTH, T = 3, 256, 64, 20

if __name__ == "__main__":
    fc = [lp_column("zipf1.1.dat" + s) for s in ("", "2", "3")]
    allk, F = merge_universe(fc)
    tF = [float((f ** P).sum()) for f in F]
    t2 = float((F[0] * F[1]).sum())
    t3 = float((F[0] * F[1] * F[2]).sum())
    print(f"union keys = {len(allk)}")
    e = {f"F3[{i}]": [] for i in range(3)}
    e["join(0,1)"] = []; e["star"] = []
    for t in range(T):
        rng = np.random.default_rng(6000 + t)
        fam = ccs.stored_family(F, P, W, rng, d=DEPTH)
        tasks = ccs.family_tasks(fam, F, P)
        for i in range(3):
            e[f"F3[{i}]"].append(abs(ccs.median(tasks[f"F_p[{i}]"]) - tF[i]) / tF[i])
        e["join(0,1)"].append(abs(ccs.median(tasks["join(0,1)"]) - t2) / t2)
        e["star"].append(abs(ccs.median(tasks["star"]) - t3) / t3)
    truth = {"F3[0]": tF[0], "F3[1]": tF[1], "F3[2]": tF[2], "join(0,1)": t2, "star": t3}
    for kk, v in e.items():
        v = np.array(v)
        print(f"{kk:10s} truth={truth[kk]:.5g} mean={v.mean():.4g} med={np.median(v):.4g}")
    mb = 3 * W * DEPTH * 16 / 1e6
    print(f"memory: 3 sketches = {mb:.2f} MB; exact F3 array over the union "
          f"= {len(allk)*4/1e6:.1f} MB")
    print("EXP-JOINT DONE")
