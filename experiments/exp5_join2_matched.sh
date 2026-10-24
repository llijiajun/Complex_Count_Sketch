#!/bin/bash
# Matched-budget JoinSketch baseline for real 2-joins (paper Table 7 reference):
# JoinSketch receives 131,072 counters (21845 x 6), the same counter count as
# byte-matched Count Sketch (w=2048) / CCS (w=1024 complex).
#
# ARTIFACT NOTE (vs the original paper_code/experiments/exp5_join2_matched.sh):
# the drivers read plain-text column files at ../datasets/... relative to the
# run directory.  The artifact bundles those columns gzipped under data/, so
# this script materialises exactly the six needed files into
# cpp/join2/datasets/ first (gunzip from the bundle, ~450 MB temporary), then
# builds and runs.  Pass --keep to leave the materialised datasets in place.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="$(cd "$HERE/.." && pwd)"
CPP="$ART/cpp"
BIN="$CPP/join2/matched_bin"
RUN="$CPP/join2/run_matched"
DS="$CPP/join2/datasets"
KEEP=0
[ "$1" = "--keep" ] && KEEP=1

mkdir -p "$BIN" "$RUN" "$DS/ML/ratings" "$DS/ML/tags" "$DS/Airlines"
echo "# materialising datasets from the bundle (data/ -> cpp/join2/datasets)"
for f in ML/ratings/userId ML/ratings/movieId ML/tags/userId ML/tags/movieId \
         Airlines/Origin Airlines/Dest Airlines/CRSArrTime Airlines/CRSDepTime; do
    gunzip -c "$ART/data/$f.gz" > "$DS/$f"
done

echo "# building (g++ -O2 -std=c++11)"
g++ -O2 -std=c++11 -I "$CPP/common" "$CPP/join2/MLjoin2_matched.cpp" -o "$BIN/mljoin2m"
g++ -O2 -std=c++11 -I "$CPP/common" "$CPP/join2/ALjoin2_matched.cpp" -o "$BIN/aljoin2m"

echo "# running"
( cd "$RUN" && "$BIN/mljoin2m" > ML2join_matched.out 2>&1 && "$BIN/aljoin2m" > AL2join_matched.out 2>&1 )
cat "$RUN/ML2join_matched.txt" "$RUN/Airlines2join_matched.txt"

if [ "$KEEP" = 0 ]; then
    rm -rf "$DS"
    echo "# (temporary datasets removed; pass --keep to retain them)"
fi
