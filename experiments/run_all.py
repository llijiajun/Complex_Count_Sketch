r"""One-click runner for the CCS paper artifact's experiment suite.

Usage:
    python3 experiments/run_all.py                 # run everything, then verify
    python3 experiments/run_all.py exp9 kas        # selected steps
    python3 experiments/run_all.py --verify        # verify bundled logs only
    python3 experiments/run_all.py --list          # list steps and runtimes

Every step writes its full output to results/<step>.out (created on demand)
and is then checked against the archived producer log in
results/experiments/<step>.txt.  Checks are verbatim-line containment by
default; steps whose absolute values are environment-dependent carry the
original producers' own weaker checks (noted per step below and in the
module docstring of each script).

Runtime (this machine, py311): the full suite is ~2h, dominated by exp9
(~60 min) and exp13 (~25 min).  Tables 2-5 and the two figures are NOT part
of this suite -- they live in tables/ (see tables/README).
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
RES = os.path.join(ART, "results")
ARCH = os.path.join(RES, "experiments")
PY = sys.executable

# step name -> (script, expected archive filename or None, runtime hint)
STEPS = {
    "exp2":  ("exp2_kasandr_3star.py",      "exp2_kasandr_3star.expected.txt", "~1 min"),
    "exp3b": ("exp3b_aggregators.py",       "exp3b_aggregators.txt",        "~4 min"),
    "exp4":  ("exp4_equal_budget.py",       "exp4_equal_budget.expected.txt", "~3 min"),
    "exp5":  ("exp5_join2_matched.sh",      "exp5_join2_matched.expected.txt", "~6 min (g++)"),
    "exp6":  ("exp6_ganguly_compare.py",    "exp6_ganguly_compare.txt",     "~2 min"),
    "exp7":  ("exp7_update_bench.sh",       "exp7_update_bench.txt",        "~1 min (g++)"),
    "exp8":  ("exp8_fp_ci_sweep.py",        "exp8_fp_ci_sweep.txt",         "~6 min"),
    "exp9":  ("exp9_pstar_baselines.py",    None,                           "~60 min"),
    "exp11": ("exp11_pattern_check.py",     "exp11_pattern_check.txt",      "~1 min"),
    "exp12": ("exp12_real_multistar.py",    "exp12_real_multistar.txt",     "~6 min"),
    "exp13": ("exp13_misalignment_sweep.py", "exp13_misalignment_sweep.txt", "~25 min"),
    "exp14": ("exp14_merge_delete.py",      "exp14_merge_delete.txt",       "~6 min"),
    "exp15": ("exp15_planner_rank.py",      "exp15_planner_rank.txt",       "~3 min"),
    "exp16": ("exp16_real_solutions.py",    "exp16_real_solutions.txt",     "~2 min"),
    "exp17": ("exp17_ganguly_attribution.py", "exp17_ganguly_attribution.txt", "~10 min"),
    "exp18": ("exp18_joint_workload.py",    "exp18_joint_workload.txt",     "~3 min"),
}

# AMS reference values (paper Table 2, synthetic rows) for the exp4
# directional checks -- the w=100 uniform regime is very high-variance and
# its absolute values are numpy-version dependent (see exp4's docstring).
AMS = {("uniform", 3): 8.3e-3, ("uniform", 4): 2.4e-2, ("uniform", 5): 1.9e-2,
       ("zipf1.1", 3): 3.6e-2, ("zipf1.1", 4): 1.6e-1, ("zipf1.1", 5): 9.5e-2,
       ("zipf1.5", 3): 5.4e-3, ("zipf1.5", 4): 8.4e-3, ("zipf1.5", 5): 6.9e-2,
       ("mix", 3): 3.1e-2, ("mix", 4): 4.6e-3, ("mix", 5): 3.8e-2}


def _lines(path):
    with open(path) as fh:
        return [l.rstrip() for l in fh if l.strip() and not l.startswith("#")]


def _data_lines(path):
    """Archived lines minus non-experimental chatter:
      * runner bookkeeping markers -- the original producer's run_all
        appended one per step, in two formats ('EXP6_EXIT 0', 'E12 0');
      * the original machine's numpy deprecation warnings (exp15's
        np.fromstring block, exp16's np.math block): the ports replaced
        both deprecated aliases (loaders accessor / math module), so the
        warnings cannot and should not reproduce.
    The scripts' own end markers ('EXP11_DONE', 'ALL_DONE') ARE experiment
    output and are kept.
    """
    out = []
    for l in _lines(path):
        if re.match(r"EXP\d+_EXIT\b", l) or re.match(r"E\d+ \d+$", l):
            continue
        if "Warning" in l or re.search(r"np\.(math|fromstring)", l):
            continue
        out.append(l)
    return out


def check_lines(step, exp_path, got_path):
    """Default: every archived line appears verbatim in the new output."""
    miss = [l for l in _data_lines(exp_path) if l not in set(_lines(got_path))]
    if miss:
        for l in miss[:5]:
            print(f"    missing: {l}")
    return not miss


def check_exp5(step, exp_path, got_path):
    """JoinSketch matched-budget: compare the driver's RESULT lines only.

    The archive interleaves the original machine's g++ warnings (absolute
    paper_code paths) with the results; the warnings are build chatter that
    legitimately differs per build location, the result rows are the
    experiment.  Everything else in the archive is ignored.
    """
    def results(path):
        return [l for l in _lines(path)
                if l.startswith(("userId", "movieId", "result:"))]
    arch, got = results(exp_path), results(got_path)
    miss = [l for l in arch if l not in set(got)]
    if miss:
        for l in miss[:5]:
            print(f"    missing: {l}")
    extra = [l for l in got if l not in set(arch)]
    if extra:
        for l in extra[:5]:
            print(f"    unexpected: {l}")
    return not miss and not extra


def check_exp4(step, exp_path, got_path):
    """Directional claims (the original producer's own verification)."""
    gotv = {}
    for l in _lines(got_path):
        m = re.match(r"CCS w=100 d=6 (\S+) p=(\d): mean=(\S+)", l)
        if m:
            gotv[(m.group(1), int(m.group(2)))] = float(m.group(3))
    ok = True
    for (dist, p), ams_v in sorted(AMS.items()):
        v = gotv.get((dist, p))
        if v is None:
            ok = False
            print(f"    missing row {dist} p={p}")
            continue
        good = v > ams_v if dist == "uniform" else v < max(ams_v, 2.5e-2)
        if not good:
            ok = False
            print(f"    FAIL {dist} p={p}: {v:.3e} vs AMS {ams_v:.3e}")
    return ok


def check_exp7(step, exp_path, got_path):
    """Throughput is machine-dependent: check the ordering, not the values."""
    rates = {}
    for l in _lines(got_path):
        m = re.match(r"(CountSketch|ComplexCountSketch|AMS \(sample-count\))\s*:\s*([\d.]+) M", l)
        if m:
            rates[m.group(1)] = float(m.group(2))
    ok = len(rates) == 3
    if ok:
        ok = rates["CountSketch"] > rates["ComplexCountSketch"] > rates["AMS (sample-count)"]
    print(f"    rates (this machine): {rates}")
    return ok


def check_exp9(step, exp_path, got_path):
    """Numeric comparison keyed by the (unique) true join size.

    The archived synth logs' combo tags were mangled when the logs were
    captured ('uniform'->'unifor', and zipf1.1/zipf1.5 both truncate to
    'zipf1.'), so tags cannot key the comparison -- synth2 and synth3 share
    a tag, as do synth4 and synth5.  The true join size is unique per
    construct and is printed identically on both sides, so it keys the
    match instead.  The per-combo archives live in exp9_synth{2..5}.txt /
    exp9_ml.txt / exp9_kas.txt; exp9_pstar_baselines_synth01.txt holds the
    first two combos.  exp_path is unused (the archives are read directly).
    """
    fields = r"(truth|CCS mean|peredge mean|med|neg|bias|" \
             r"sharedsign mean|cubes mean)=([\d.eE+-]+)"

    def parse(line):
        key, vals = None, {}
        for m in re.finditer(fields, line):
            k, v = m.group(1), float(m.group(2))
            if k == "truth":
                key = f"{v:.6g}"
            else:  # 'med'/'neg' repeat per estimator: keep them all
                vals.setdefault(k, []).append(v)
        return key, vals

    arch, got = {}, {}
    for name in ("exp9_pstar_baselines_synth01.txt", "exp9_synth2.txt",
                 "exp9_synth3.txt", "exp9_synth4.txt", "exp9_synth5.txt",
                 "exp9_ml.txt", "exp9_kas.txt"):
        for l in _lines(os.path.join(ARCH, name)):
            if l.startswith("["):
                k, v = parse(l)
                if k is not None:
                    arch[k] = v
    for l in _lines(got_path):
        if l.startswith("["):
            k, v = parse(l)
            if k is not None:
                got[k] = v
    ok = True
    for k in sorted(arch):
        g = got.get(k)
        if g is None:
            ok = False
            print(f"    missing construct: truth={k}")
            continue
        for f, vs in arch[k].items():
            gs = g.get(f)
            if gs is None or len(gs) != len(vs) or any(
                    abs(a - b) > 2e-3 * max(abs(b), 1e-12)
                    for a, b in zip(gs, vs)):
                ok = False
                print(f"    truth={k} {f}: archived {vs} got {gs}")
    for k in sorted(set(got) - set(arch)):
        ok = False
        print(f"    unexpected construct: truth={k}")
    return ok


CHECKERS = {"exp4": check_exp4, "exp5": check_exp5, "exp7": check_exp7,
            "exp9": check_exp9}


def run_step(name):
    script, _arch, hint = STEPS[name]
    print("=" * 70)
    print(f"RUN {script}  ({hint})", flush=True)
    out = os.path.join(RES, name + ".out")
    t0 = time.time()
    if script.endswith(".sh"):
        r = subprocess.run(["bash", os.path.join(HERE, script)],
                           stdout=open(out, "w"), stderr=subprocess.STDOUT)
    else:
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        r = subprocess.run([PY, os.path.join(HERE, script)],
                           stdout=open(out, "w"), stderr=subprocess.STDOUT,
                           env=env, cwd=ART)
    print(f"DONE {script}  exit {r.returncode}  ({time.time()-t0:.0f}s)",
          flush=True)
    return r.returncode == 0


def verify(names):
    print("=" * 70)
    print("VERIFY against archived producer logs", flush=True)
    ok_all = True
    for name in names:
        script, arch_name, _ = STEPS[name]
        got_path = os.path.join(RES, name + ".out")
        if not os.path.exists(got_path):
            print(f"  {name}: not run in this invocation")
            continue
        checker = CHECKERS.get(name, check_lines)
        if arch_name is None and checker is check_lines:
            print(f"  {name}: no archived log bundled (see experiments/README)")
            continue
        exp_path = os.path.join(ARCH, arch_name) if arch_name else None
        if checker is check_lines and (exp_path is None
                                       or not os.path.exists(exp_path)):
            print(f"  {name}: archived log {arch_name} not found")
            continue
        ok = checker(name, exp_path, got_path)
        print(f"  {name}: {'MATCH' if ok else 'MISMATCH'}", flush=True)
        ok_all &= ok
    print("OVERALL:", "ALL CHECKS PASS" if ok_all else "SOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        for k, (s, a, h) in STEPS.items():
            print(f"  {k:6s} {s:32s} {h}")
        sys.exit(0)
    if "--verify" in args:
        sys.exit(0 if verify(list(STEPS)) else 1)
    wanted = [a for a in args if not a.startswith("--")] or list(STEPS)
    bad = [n for n in wanted if n not in STEPS]
    if bad:
        sys.exit(f"unknown step(s): {bad} (see --list)")
    ok = True
    for name in wanted:
        ok &= run_step(name)
    verify(wanted)
    sys.exit(0 if ok else 1)
