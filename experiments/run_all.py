"""Run every experiment in this directory; logs to ../results/.

Usage:
    python run_all.py            # all experiments (hours)
    python run_all.py exp01      # one experiment
    python run_all.py quick      # the fast subset (~15 min)
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

QUICK = ["exp01_fp_skew", "exp06_pattern_check", "exp07_real_solutions",
         "exp10_merge_delete", "exp13_joint_workload"]
ALL = sorted(f[:-3] for f in os.listdir(HERE)
             if f.startswith("exp") and f.endswith(".py"))

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = QUICK if arg == "quick" else ([arg] if arg != "all" else ALL)
    for name in targets:
        print(f"===== {name} =====", flush=True)
        out = open(os.path.join(RES, name + ".out"), "w")
        t0 = time.time()
        rc = subprocess.call([sys.executable, name + ".py"], cwd=HERE,
                             stdout=out, stderr=subprocess.STDOUT)
        out.close()
        print(f"  exit {rc} in {time.time()-t0:.0f}s", flush=True)
