"""loaders.py -- transparent data loading for the CCS paper artifact.

Bundled data (data/, 409 MB, gzip-compressed column files)
----------------------------------------------------------
lp/{uniform,zipf1.1,zipf1.5}.dat{,2,3}.gz       10^7-row synthetic columns
lp/uniform+zipf1.1.dat.gz                       equal-probability mixture
ML/ratings/{userId,movieId,rating}.gz           MovieLens ratings (25.0M rows)
ML/tags/{userId,movieId}.gz                     MovieLens tags (1.09M rows)
airlines/*.gz                                   Airlines (10.0M rows, 10 cols)
kasandr_essential.npz                           Kasandr train (users/cats/times)
kasandr_extra_columns.npz                       offerid/countrycode/merchant
                                                distinct views (exp15)
kasandr_columns/*.kc.tsv.gz                     the 7 merge-view columns

All bundled files are byte-identical to the originals used for the paper
(md5-verified; see README).  Two access patterns:

  * col(name)      -> (distinct keys, counts)        frequency-vector view
  * lines(name)    -> (distinct raw line strings, counts)
                      string view, needed by the MurmurHash-based C++-
                      compatible estimators (the C++ drivers hash the raw
                      line text, not the parsed integer).
"""
import gzip
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

LP_COLS = ("uniform", "zipf1.1", "zipf1.5", "uniform+zipf1.1")
LP_SUFS = ("", "2", "3")
ML_RATINGS = ("userId", "movieId", "rating")
ML_TAGS = ("userId", "movieId")
AIRLINES = ("CRSArrTime", "CRSDepTime", "DayofMonth", "DayOfWeek", "DepDelay",
            "Dest", "Distance", "Month", "Origin", "UniqueCarrier")


def _read_tokens(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as fh:
        toks = fh.read().split()
    return toks


def _parse(path):
    """(values, is_string): int64 array when the column is integral."""
    toks = _read_tokens(path)
    try:
        return np.array(toks, dtype=np.int64), False
    except (ValueError, TypeError):
        return np.array([t.decode() for t in toks], dtype=object), True


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(DATA, path)


def lines(path):
    """(distinct raw line strings as bytes, counts) of a bundled column."""
    p = _resolve(path)
    toks = _read_tokens(p)
    arr = np.array(toks, dtype=object)          # bytes tokens, C++-identical
    k, c = np.unique(arr, return_counts=True)
    return k, c


def col(path):
    """(distinct keys, counts): int64 keys when integral, else str keys."""
    p = _resolve(path)
    a, _ = _parse(p)
    return np.unique(a, return_counts=True)


def raw(path):
    """Raw value array of a bundled column (int64 when integral)."""
    a, _ = _parse(_resolve(path))
    return a


def dense(k, c, universe):
    """Dense float frequency vector over `universe` (a sorted key array)."""
    f = np.zeros(len(universe))
    f[np.searchsorted(universe, k)] = c
    return f


def merge_universe(freq_cols):
    """Union universe + dense float frequency vectors for (k, c) columns."""
    allk = np.unique(np.concatenate([k for k, _ in freq_cols]))
    F = [dense(k, c, allk) for k, c in freq_cols]
    return allk, F


# ---- named column accessors ------------------------------------------------- #

def lp_column(name):
    """(keys, counts) of a bundled Lp synthetic column, e.g. 'zipf1.1.dat2'."""
    return col(os.path.join("lp", name + ".gz"))


def lp_lines(name):
    """(distinct raw lines, counts) of a bundled Lp column (hash view)."""
    return lines(os.path.join("lp", name + ".gz"))


def ml_ratings(key):
    return col(os.path.join("ML", "ratings", key + ".gz"))


def ml_ratings_lines(key):
    return lines(os.path.join("ML", "ratings", key + ".gz"))


def ml_tags(key):
    return col(os.path.join("ML", "tags", key + ".gz"))


def airlines(key):
    return col(os.path.join("airlines", key + ".gz"))


def airlines_lines(key):
    return lines(os.path.join("airlines", key + ".gz"))


def kasandr():
    """Kasandr essential columns: (users, cats, times, n_users), row-aligned.

    Extracted from train_de.csv exactly as exp2_kasandr_3star.load() does
    (user/category ids by first appearance, times as YYYYMMDD integers).
    """
    z = np.load(os.path.join(DATA, "kasandr_essential.npz"))
    return z["users"], z["cats"], z["times"], int(z["n_users"])


KAS_EXTRA = ("offerid", "countrycode", "merchant")


def kasandr_extra(name):
    """(distinct keys, counts) of an extra Kasandr field, for exp15.

    Extracted from train_de.csv columns offerid / countrycode / merchant.
    The original exp15 keyed these string fields with Python's built-in
    hash(), which is salted per process (PYTHONHASHSEED) and therefore not
    reproducible; the artifact keys them with zlib.crc32 instead.  Only the
    distinct-key multiplicities matter for F_3 and for the CCS/cubes
    estimators, and those are identical either way.
    """
    z = np.load(os.path.join(DATA, "kasandr_extra_columns.npz"))
    return z[name + "_k"], z[name + "_c"]


KAS_COLS = ("rating", "utcdate", "merchant", "category", "countrycode",
            "offerid", "userid")


def kasandr_column(name):
    """(distinct keys as bytes, counts, N) of a bundled kasandr merge column.

    The C++ Table 2 driver (kasandr.cpp) hashes each raw line of the merge
    column; the CCS/AMS estimators and the exact truth depend only on the
    distinct keys and their multiplicities, which is what is bundled
    (data/kasandr_columns/<name>.kc.tsv.gz, verified against the archived
    true_merge{3,4,5}.csv).  N is the original row count (needed by AMS).
    """
    path = os.path.join(DATA, "kasandr_columns", name + ".kc.tsv.gz")
    keys, counts = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            k, c = line.rstrip("\n").split("\t")
            keys.append(k.encode())
            counts.append(int(c))
    return np.array(keys, dtype=object), np.array(counts), int(sum(counts))


def star_truth(freq_cols):
    """Exact p-star join size: sum over common keys of the count product."""
    common = freq_cols[0][0]
    for k, _ in freq_cols[1:]:
        common = np.intersect1d(common, k, assume_unique=True)
    truth = np.ones(len(common))
    for k, c in freq_cols:
        truth *= c[np.searchsorted(k, common)]
    return float(truth.sum())
