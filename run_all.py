r"""run_all.py -- artifact-wide smoke test (fast) and full-run guide.

Usage:
    python3 run_all.py            # smoke: every component at reduced settings
    python3 run_all.py --full     # print the full-protocol commands (no run)

The smoke test exercises every import path, data loader and estimator at a
small fraction of the paper protocol (minutes, not hours) and checks the
archive parsers.  It is NOT a reproduction of the paper numbers -- for that,
run each table/experiment at its default protocol (see --full).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FULL = {
    "Table 2":  "python3 tables/table2_lpnorm.py            (~10 min)",
    "Table 2 audit": "python3 tables/table2_archive_audit.py (seconds)",
    "Table 3":  "python3 tables/table3_tmccs.py             (~4 min)",
    "Table 4":  "python3 tables/table4_pstar.py             (~45 min)",
    "Table 5":  "python3 tables/table5_realccs.py           (~6 min)",
    "Figures":  "python3 tables/figures.py --recompute      (~2 min)",
    "Experiments": "python3 experiments/run_all.py           (~2.5 h)",
    "exp5 (C++)": "bash experiments/exp5_join2_matched.sh   (~6 min, g++)",
    "exp7 (C++)": "bash experiments/exp7_update_bench.sh    (~1 min, g++)",
}

SMOKE = [
    ("imports + data smoke", [PY, "-c",
        "import sys; sys.path.insert(0, '.');"
        "import ccs, loaders;"
        "print('ccs/loaders import OK');"
        "k, c = loaders.lp_column('uniform.dat');"
        "print('lp_column uniform.dat:', len(k), 'distinct keys');"
        "u, ct, t, n = loaders.kasandr();"
        "print('kasandr:', len(u), 'rows,', n, 'users');"
        "k2, c2 = loaders.kasandr_extra('merchant');"
        "print('kasandr merchant:', len(k2), 'distinct')"]),
    ("Table 3 smoke (60 runs)", [PY, "tables/table3_tmccs.py", "60"]),
    ("Table 4 smoke (T=4)", [PY, "tables/table4_pstar.py", "--trials", "4"]),
    ("Table 5 smoke (T=4)", [PY, "tables/table5_realccs.py", "4"]),
    ("figures (archive mode)", [PY, "tables/figures.py"]),
    ("exp11 pattern check", [PY, "experiments/exp11_pattern_check.py"]),
]


def main():
    if "--full" in sys.argv:
        print("# full-protocol commands (paper numbers):")
        for k, v in FULL.items():
            print(f"  {k:16s} {v}")
        return 0
    ok = True
    for name, cmd in SMOKE:
        print("=" * 70)
        print("SMOKE", name, flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE)
        print(f"-- {name}: exit {r.returncode} ({time.time()-t0:.0f}s)",
              flush=True)
        ok &= (r.returncode == 0)
    print("=" * 70)
    print("SMOKE OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
