#!/bin/bash
# Update-throughput benchmark (elements/second): Count Sketch vs CCS vs AMS
# (sample-count scans width x AMS_num positions per update). Same item stream,
# same machine; 10^6 inserts each at eps=0.1.  Synthetic stream: no dataset
# files needed.
#
# ARTIFACT NOTE (vs the original paper_code/experiments/exp7_update_bench.sh):
# paths rebased onto the artifact layout; otherwise verbatim.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="$(cd "$HERE/.." && pwd)"
CPP="$ART/cpp"
mkdir -p "$CPP/bench"
g++ -O2 -std=c++14 -I "$CPP/common" "$CPP/bench/bench_update.cpp" -o "$CPP/bench/bench_update"
"$CPP/bench/bench_update"
