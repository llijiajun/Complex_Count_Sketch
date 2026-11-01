"""loaders.py -- transparent data loading for the CCS release.

All bundled column files are gzip-compressed whitespace-separated integer
files (one value per row, as released by the original datasets).  `int_col`
returns (distinct keys, counts); `raw` returns the raw value array.
"""
import gzip
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _read_ints(path):
    op = gzip.open if path.endswith(".gz") else open
    raw = op(path, "rb").read()
    a = np.fromstring(raw, dtype=np.int64, sep=" ")
    if len(a) == 0:                               # non-integer tokens
        s = np.array(raw.split(), dtype=object)
        return s
    return a


def col(path):
    """(distinct keys, counts) of a bundled column file."""
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    a = _read_ints(p)
    if a.dtype == object:
        return np.unique(a, return_counts=True)
    return np.unique(a, return_counts=True)


def raw(path):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    return _read_ints(p)


def dense(k, c, universe):
    """Dense frequency vector over `universe` (a sorted key array)."""
    f = np.zeros(len(universe))
    f[np.searchsorted(universe, k)] = c
    return f


def kasandr():
    """Kasandr essential columns: (users, cats, times, n_users), each row aligned."""
    z = np.load(os.path.join(DATA, "kasandr_essential.npz"))
    return z["users"], z["cats"], z["times"], int(z["n_users"])


def lp_column(name):
    """(distinct keys, counts) of a bundled Lp synthetic column, e.g. 'zipf1.1.dat2'."""
    return col(os.path.join("lp", name + ".gz"))


def merge_universe(freq_cols):
    """Union universe + dense float frequency vectors for a list of (k, c)."""
    allk = np.unique(np.concatenate([k for k, _ in freq_cols]))
    F = [dense(k, c, allk) for k, c in freq_cols]
    return allk, F
