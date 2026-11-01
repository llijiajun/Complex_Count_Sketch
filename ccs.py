"""ccs.py -- Core library for the Complex Count Sketch (CCS) experiments.

Everything the experiment scripts need to build and query the sketches of the
paper "Complex Count Sketch: Mergeable Frequency-Moment and p-Star Join Size
Estimation", in one place:

  * star_rows          -- the p-star / F_p row estimates of a p-column CCS
                          (shared h and g across columns, all-ones exponents)
  * f_p_rows           -- single-column F_p row estimates (same sketch, p=order)
  * join_rows          -- 2-join row estimates from two stored sketches
                          (conjugation: exponents +1, -1; answers at any order)
  * peredge_star       -- the real-valued per-edge Rademacher star estimator
                          (order-2 instance of Prop. 4.7; unbiased)
  * sharedsign_star    -- one +-1 hash shared by all columns (biased)
  * cubes_real         -- cubes of one real Count Sketch over the merged stream
  * tmccs              -- Tukey-truncated median over |Im(X_j)|
  * g_moments, exact_power_expectation
                       -- exact expectation of the power-sum estimator for a
                          real multiplier (used to verify unbiasedness)

Conventions
-----------
F          list of dense frequency vectors (float64), one per column, all over
           the same key universe of size U.
p          root order (sketch order).
w, d       width (buckets per row) and depth (number of rows).
rng        numpy Generator; each row uses independent draws from it.
"""
import math
import numpy as np


# --------------------------------------------------------------------------- #
# Complex Count Sketch
# --------------------------------------------------------------------------- #

def star_rows(F, p, w, rng, d=64):
    """Row estimates of the p-star join (and of every column's F_p).

    Builds one order-p CCS per column with SHARED h and g (the deployment
    requirement: the same key carries the same root in every table), then
    multiplies the counters element-wise across columns.

    Returns (re, im): per-row Re(sum_s prod_i C_i[j,s]) and |Im(...)|.
    """
    U = len(F[0])
    re = np.empty(d); im = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        gr, gi = np.cos(ang), np.sin(ang)
        pr, pi = np.ones(w), np.zeros(w)
        for f in F:                                  # element-wise product
            cr = np.bincount(h, weights=f * gr, minlength=w)
            ci = np.bincount(h, weights=f * gi, minlength=w)
            pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
        re[j] = pr.sum()
        im[j] = abs(pi.sum())
    return re, im


def f_p_rows(freq, p, w, rng, d=64):
    """Single-column F_p row estimates: X_j = sum_s C[j,s]^p (real part)."""
    U = len(freq)
    re = np.empty(d); im = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        gr, gi = np.cos(ang), np.sin(ang)
        cr = np.bincount(h, weights=freq * gr, minlength=w)
        ci = np.bincount(h, weights=freq * gi, minlength=w)
        X = (cr + 1j * ci) ** p
        re[j] = X.real.sum()
        im[j] = abs(X.imag.sum())
    return re, im


def join_rows(CR, CI, CS, SI, p):
    """2-join row estimates from two STORED order-p sketches (conjugation).

    sum_s C^R_s * conj(C^S_s): exponents +1 and -1 sum to 0 mod p, so the same
    sketches answer the 2-join at every order.  CR/CI are the real/imaginary
    parts of the first sketch's counters (d x w); CS/SI of the second.
    """
    d = CR.shape[0]
    return ((CR * CS + CI * SI).sum(axis=1))


def stored_sketch(freq, p, w, rng, d=64):
    """One stored order-p CCS: returns (CR, CI), the d x w real/imag parts."""
    U = len(freq)
    CR = np.empty((d, w)); CI = np.empty((d, w))
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        CR[j] = np.bincount(h, weights=freq * np.cos(ang), minlength=w)
        CI[j] = np.bincount(h, weights=freq * np.sin(ang), minlength=w)
    return CR, CI


# --------------------------------------------------------------------------- #
# Real-valued star baselines (Section 5 baselines)
# --------------------------------------------------------------------------- #

def peredge_star(F, w, rng, d=64):
    """Per-edge Rademacher 3+-star at byte-matched width 2w (unbiased).

    One independent +-1 sign per edge of the star; column t carries the
    product of its incident edge signs, so the diagonal contributes
    prod_e s_e^2 = 1 (order-2 instance of Prop. 4.7).
    """
    U = len(F[0]); W2 = 2 * w; k = len(F)
    est = np.empty(d)
    for j in range(d):
        h = rng.integers(0, W2, size=U)
        signs = []
        for i in range(1, k):                        # edges (1, i+1)
            signs.append(rng.integers(0, 2, size=U) * 2 - 1)
        col_signs = [np.prod(signs, axis=0)] + signs  # col1 = product of edges
        cs = [np.bincount(h, weights=f * sg, minlength=W2)
              for f, sg in zip(F, col_signs)]
        est[j] = float(np.prod(cs, axis=0).sum())
    return est


def sharedsign_star(F, w, rng, d=64):
    """One +-1 hash shared by all columns: E[g^p] = E[g] = 0 for odd p (biased)."""
    U = len(F[0])
    est = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        s = rng.integers(0, 2, size=U) * 2 - 1
        cs = [np.bincount(h, weights=f * s, minlength=w) for f in F]
        est[j] = float(np.prod(cs, axis=0).sum())
    return est


def cubes_real(F, w, rng, d=64, p=3):
    """Sum of p-th powers of one real Count Sketch over the merged stream."""
    U = len(F[0])
    Fsum = np.sum(F, axis=0) if isinstance(F, list) else F.sum(axis=0)
    est = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        s = rng.integers(0, 2, size=U) * 2 - 1
        c = np.bincount(h, weights=Fsum * s, minlength=w)
        est[j] = float((c ** p).sum())
    return est


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def tmccs(re, im, keep_frac=0.5):
    """Tukey-truncated median: drop rows with |Im| > Q3 + 1.5 IQR (>=half kept)."""
    im = np.asarray(im)
    q1, q3 = np.percentile(im, [25, 75])
    keep = np.nonzero(im <= q3 + 1.5 * (q3 - q1))[0]
    if len(keep) < int(np.ceil(keep_frac * len(re))):
        keep = np.argsort(im)[: len(re) // 2]
    return float(np.median(re[keep]))


def median(re):
    return float(np.median(re))


def mean(re):
    return float(np.mean(re))


def med_of_means(re, groups=8):
    return float(np.median([g.mean() for g in np.array_split(re, groups)]))


# --------------------------------------------------------------------------- #
# Exact verification utilities (real multipliers)
# --------------------------------------------------------------------------- #

def g_moments(support, weights, upto):
    """mu_m for m = 0..upto of a real multiplier (mu_0 = 1 by convention)."""
    mu = np.ones(upto + 1)
    for m in range(1, upto + 1):
        mu[m] = float((weights * support ** m).sum())
    mu[0] = 1.0
    return mu


def exact_power_expectation(p, freq, mu):
    """E[(sum_alpha f_alpha S_alpha)^p], S_alpha iid with moments mu[0..p].

    Dynamic program over the keys; exact to machine precision (no sampling).
    The power-sum estimator is unbiased iff this equals sum_alpha f_alpha^p.
    """
    D = len(freq)
    dp = np.zeros(p + 1); dp[0] = 1.0
    for a in range(D):
        f = freq[a]
        if f == 0:
            continue
        new = np.zeros(p + 1)
        for e in range(p + 1):
            term, acc = 1.0, 0.0
            for n in range(0, e + 1):
                if n > 0:
                    term *= f
                acc += dp[e - n] * term * mu[n]
            new[e] = acc
        dp = new
    return float(dp[p])


def exact_solve_real_multiplier(p, support):
    """Exact linear solve of the moment conditions on a fixed support.

    Equations: sum w = 1; odd moments m = 1,3,..,p-2 equal 0; mu_p = 1.
    Raises if the support admits no strictly positive solution.
    """
    eqs = [np.ones(len(support))] + [support ** m for m in range(1, p, 2)] \
        + [support ** p]
    A = np.vstack(eqs)
    b = np.zeros(len(eqs)); b[0] = 1.0; b[-1] = 1.0
    w = np.linalg.solve(A, b)
    assert np.all(w > 1e-9), f"support {support} not strictly feasible"
    return w


def stored_family(F, p, w, rng, d=64):
    """A SHARED-hash deployment: one order-p CCS per column, all columns using
    the same h and g draws (the deployment requirement for multi-task reuse).

    F: list of dense frequency vectors.  Returns a list of (CR, CI) pairs,
    one d x w real/imaginary counter array per column.
    """
    U = len(F[0])
    out = []
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        gr, gi = np.cos(ang), np.sin(ang)
        for i, f in enumerate(F):
            if j == 0:
                out.append((np.empty((d, w)), np.empty((d, w))))
            out[i][0][j] = np.bincount(h, weights=f * gr, minlength=w)
            out[i][1][j] = np.bincount(h, weights=f * gi, minlength=w)
    return out


def family_tasks(sketches, freqs, p):
    """Answer F_p / 2-join / p-star from one shared-hash deployment.

    sketches: list of (CR, CI) from stored_family.  freqs: dense vectors
    (for truths at the call site).  Returns dict of per-row estimates:
      'F_p[i]' per column, 'join(i,j)' pairwise, 'star' full product.
    """
    out = {}
    powers = [(CR + 1j * CI) ** p for (CR, CI) in sketches]
    for i, X in enumerate(powers):
        out[f"F_p[{i}]"] = X.real.sum(axis=1)
    for i in range(len(sketches)):
        for j in range(i + 1, len(sketches)):
            CR, CI = sketches[i]
            DS, DI = sketches[j]
            out[f"join({i},{j})"] = (CR * DS + CI * DI).sum(axis=1)
    prod = np.ones_like(powers[0])
    for (CR, CI) in sketches:                     # star: each counter ONCE
        prod = prod * (CR + 1j * CI)
    out["star"] = prod.real.sum(axis=1)
    return out


# --------------------------------------------------------------------------- #
# C++-pipeline-compatible estimators (paper Table 2, bit-for-bit)
# --------------------------------------------------------------------------- #

def murmur2_32(key, seed):
    """Exact port of MurmurHash32 in cpp/common/MurmurHash.h (MurmurHash2).

    `key` is bytes (the raw line text, as the C++ drivers hash it), `seed`
    an unsigned 32-bit int.  Returns the 32-bit hash as a Python int.
    """
    if isinstance(key, str):
        key = key.encode()
    n = len(key)
    h = (seed ^ n) & _MASK
    i = 0
    while n - i >= 4:
        k = int.from_bytes(key[i:i + 4], "little")
        k = (k * _M) & _MASK
        k ^= k >> 24
        k = (k * _M) & _MASK
        h = (h * _M) & _MASK
        h ^= k
        i += 4
    rem = n - i
    if rem == 3:
        h ^= key[i + 2] << 16
        h ^= key[i + 1] << 8
        h ^= key[i]
        h = (h * _M) & _MASK
    elif rem == 2:
        h ^= key[i + 1] << 8
        h ^= key[i]
        h = (h * _M) & _MASK
    elif rem == 1:
        h ^= key[i]
        h = (h * _M) & _MASK
    h ^= h >> 13
    h = (h * _M) & _MASK
    h ^= h >> 15
    return h


def cpp_ccs_estimate(keys, counts, p, eps, gam, hash_seed, hashes=None):
    """ComplexCountSketch(p, eps, gam, hash_seed).estimate_lp() on a column.

    `keys`: list of distinct raw line strings (bytes or str).
    `counts`: their multiplicities.  Mirrors ccs.py's C++ estimator: repeated
    complex multiplication for C^p (NOT numpy's complex power), median over
    the depth rows (even depth -> mean of the two middle order statistics).

    `hashes`: optional precomputed (bucket_hash, root_hash) from
    cpp_ccs_hashes -- same values, reused across p / rounds.
    """
    width, depth = cpp_dims(p, eps, gam)
    counts = np.asarray(counts, dtype=np.float64)
    if hashes is None:
        hashes = cpp_ccs_hashes(keys, hash_seed, depth)
    hb, hr = hashes
    buckets = hb % width
    roots = hr % p
    ang = (2.0 * math.pi / p) * roots
    gr, gi = np.cos(ang), np.sin(ang)                    # (depth, m)
    row_est = np.empty(depth)
    for i in range(depth):
        cr = np.bincount(buckets[i], weights=counts * gr[i], minlength=width)
        ci = np.bincount(buckets[i], weights=counts * gi[i], minlength=width)
        a, b = cr.copy(), ci.copy()                      # C, then C^p by
        for _ in range(p - 1):                           # repeated squaring
            a, b = a * cr - b * ci, a * ci + b * cr      # (as in the C++)
        row_est[i] = a.sum()
    row_est.sort()
    if depth % 2 == 0:
        return float((row_est[depth // 2] + row_est[depth // 2 - 1]) / 2)
    return float(row_est[depth // 2])


def cpp_ams_estimate(keys, counts, N, p, eps, gam, hash_seed, seed=0):
    """AMS(eps, N, gam, hash_seed) positional sample-count estimator.

    Mirrors cpp/common/countsketch.hpp: width = ceil(1/eps^2) sampled
    positions per row, depth = ceil(ln(4/gam)) rows; for each sampled
    position the item's 32-bit hash is stored and every later position with
    the same hash increments the counter (count-after of the sampled item);
    the row estimate is N * sum_j (c^p - (c-1)^p) / width; median over rows.

    The original C++ draws positions from std::random_device (not
    reproducible); this port draws them from numpy default_rng(seed) so runs
    are deterministic -- see README "Reproducibility disclosure".
    """
    width, depth = ccs_width_ams(eps, gam)
    counts = np.asarray(counts, dtype=np.float64)
    keys = list(keys)
    # per-key 32-bit hashes for every (row, bucket) seed, as the C++ stores
    # value[i][j] = (int)MurmurHash32(str, len, hash_seed + i*j)
    row_est = np.empty(depth)
    rng = np.random.default_rng(seed)
    # cumulative counts let us sample "position -> key" and count-after
    cum = np.cumsum(counts)
    total = float(cum[-1])
    for i in range(depth):
        pos = rng.integers(0, N, size=width)
        # key at each sampled position (positions are over the N stream slots)
        ki = np.searchsorted(cum, pos, side="right")
        ki = np.clip(ki, 0, len(keys) - 1)
        # count-after: uniform over the occurrences of that key
        ck = counts[ki]
        # occurrence index of the sampled slot within its key's block
        before = cum[ki] - ck
        occ = pos - before                                # 0-based occurrence
        c_after = ck - occ
        # hash equality: two different keys collide only with prob 2^-32,
        # which the C++ also tolerates; key identity is used here.
        x = N * (np.power(c_after, p) - np.power(c_after - 1.0, p))
        row_est[i] = x.sum() / width
    row_est.sort()
    if depth % 2 == 0:
        return float((row_est[depth // 2] + row_est[depth // 2 - 1]) / 2)
    return float(row_est[depth // 2])


# --------------------------------------------------------------------------- #
# Archived C++-pipeline helpers (Table 2 bit-for-bit reproduction; used by
# tests/test_api.py and the legacy table drivers)
# --------------------------------------------------------------------------- #
_M = 0x5BD1E995

_MASK = 0xFFFFFFFF

def murmur2_32_bulk(keys, seed):
    """Vectorized MurmurHash2 (bit-exact with murmur2_32) over many keys.

    Groups the keys by byte length (each group then shares the 4-byte block
    structure) and runs the whole group through uint64 numpy arithmetic masked
    to 32 bits, which is exact because every intermediate product of two
    32-bit values fits in 64 bits.  Used by the C++-compatible estimators on
    columns with millions of distinct keys, where the scalar port is too slow.
    """
    seed = int(seed) & _MASK
    kb = [k if isinstance(k, bytes) else str(k).encode() for k in keys]
    n = len(kb)
    out = np.empty(n, dtype=np.uint32)
    groups = {}
    for idx, k in enumerate(kb):
        groups.setdefault(len(k), []).append(idx)
    for ln, idxs in groups.items():
        m = len(idxs)
        raw = b"".join(kb[i] for i in idxs)
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(m, ln)
        h = np.full(m, (seed ^ ln) & _MASK, dtype=np.uint64)
        nfull = ln // 4
        for t in range(nfull):
            b = buf[:, 4 * t:4 * t + 4].astype(np.uint64)
            k = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24))
            k = (k * _M) & _MASK
            k ^= k >> 24
            k = (k * _M) & _MASK
            h = (h * _M) & _MASK
            h ^= k
        rem = ln % 4
        if rem:
            b = buf[:, nfull * 4:].astype(np.uint64)
            if rem == 3:
                h ^= b[:, 2] << 16
                h ^= b[:, 1] << 8
            elif rem == 2:
                h ^= b[:, 1] << 8
            h ^= b[:, 0]
            h = (h * _M) & _MASK
        h ^= h >> 13
        h = (h * _M) & _MASK
        h ^= h >> 15
        out[idxs] = h.astype(np.uint32)
    return out

def cpp_dims(p, eps, gam=0.01):
    """(width, depth) of the C++ ComplexCountSketch / CountSketch / AMS."""
    width = int(math.ceil((2.0 ** p) / eps / eps))
    depth = int(math.ceil(math.log(4.0 / gam)))
    return width, depth

def ccs_width_ams(eps, gam=0.01):
    """(width, depth) of the C++ AMS sketch (width = ceil(1/eps^2))."""
    width = int(math.ceil(1.0 / eps / eps))
    depth = int(math.ceil(math.log(4.0 / gam)))
    return width, depth

def cpp_ccs_hashes(keys, seed, depth):
    """Raw MurmurHash values for every row's bucket and root hash.

    p-independent (the C++ seeds are hash_seed+i and
    hash_seed+i*i+1+CS_num), so callers that sweep p can compute these once
    and reuse them across orders -- bit-identical to recomputing per call.
    """
    kb = list(keys)
    hb = np.empty((depth, len(kb)), dtype=np.int64)
    hr = np.empty((depth, len(kb)), dtype=np.int64)
    for i in range(depth):
        hb[i] = murmur2_32_bulk(kb, seed + i).astype(np.int64)
        hr[i] = murmur2_32_bulk(kb, seed + i * i + 1 + depth).astype(np.int64)
    return hb, hr

def _hash_keys(keys, seed, width, p, depth):
    """Per-key (bucket, root) for every row, mirroring the C++ Insert.

    C++: index = MurmurHash32(str, len, hash_seed + i) % width
         g     = MurmurHash32(str, len, hash_seed + i*i + 1 + CS_num) % p
    """
    hb, hr = cpp_ccs_hashes(keys, seed, depth)
    return hb % width, hr % p

def zipf_freq(D, N, alpha, seed):
    """make_freq of 03_tmccs_estimators.py (Table 3 / exp3 / exp3b protocol).

    alpha = 0 is uniform; otherwise truncated Zipf(alpha) over D keys.
    """
    rng = np.random.default_rng(seed)
    if alpha == 0.0:
        items = rng.integers(0, D, size=N)
    else:
        w = np.arange(1, D + 1, dtype=np.float64) ** (-alpha)
        w /= w.sum()
        items = rng.choice(D, size=N, p=w)
    return np.bincount(items, minlength=D).astype(np.float64)

def ccs_rows_batch(freq, p, w, d, seed):
    """ccs_row_stats of 03_tmccs_estimators.py (Table 3 producer).

    Batch RNG draws: buckets (d, m) then roots (d, m), m = #nonzero keys.
    Returns (re_ests, im_row): Re/|Im| of sum_s C[j, s]^p per row.
    """
    rng = np.random.default_rng(seed)
    items = np.nonzero(freq)[0]
    f = freq[items]
    m = len(items)
    buckets = rng.integers(0, w, size=(d, m))
    roots = rng.integers(0, p, size=(d, m))
    ang = 2.0 * np.pi * roots / p
    gre, gim = np.cos(ang), np.sin(ang)
    re_ests = np.empty(d)
    im_row = np.empty(d)
    for j in range(d):
        c = (np.bincount(buckets[j], weights=f * gre[j], minlength=w)
             + 1j * np.bincount(buckets[j], weights=f * gim[j], minlength=w))
        s = (c ** p).sum()
        re_ests[j] = s.real
        im_row[j] = abs(s.imag)
    return re_ests, im_row

def fp_rows_interleaved(freq, p, w, rng, d=64):
    """Per-row interleaved F_p estimator (exp8 / exp15 / exp04 protocol).

    One rng, per row j: h (U values) then g (p-valued); returns re, im.
    """
    U = len(freq)
    re = np.empty(d)
    im = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        cr = np.bincount(h, weights=freq * np.cos(ang), minlength=w)
        ci = np.bincount(h, weights=freq * np.sin(ang), minlength=w)
        X = (cr + 1j * ci) ** p
        re[j] = X.real.sum()
        im[j] = abs(X.imag.sum())
    return re, im

def cs2_rows(R, S, w, rng, d=64):
    """Count Sketch (order-2 specialist) 2-join estimator, exp1 protocol.

    R, S are (positions, counts) over a shared universe: per row j, one h
    (U values) and one shared sign s (U values); the row estimate is
    sum_s CR_s * CS_s (the FFT/inner-product form of Fast-AGMS).
    """
    allk = np.unique(np.concatenate([R[0], S[0]]))
    posR = np.searchsorted(allk, R[0])
    posS = np.searchsorted(allk, S[0])
    U = len(allk)
    rej = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        s = rng.integers(0, 2, size=U) * 2 - 1
        crR = np.bincount(h[posR], weights=R[1] * s[posR], minlength=w)
        crS = np.bincount(h[posS], weights=S[1] * s[posS], minlength=w)
        rej[j] = float(np.sum(crR * crS))
    return rej

def ccs_cross2_rows(R, S, p, w, rng, d=64):
    """Order-p CCS 2-join estimator, exp1 protocol (one root per key)."""
    allk = np.unique(np.concatenate([R[0], S[0]]))
    posR = np.searchsorted(allk, R[0])
    posS = np.searchsorted(allk, S[0])
    U = len(allk)
    rej = np.empty(d)
    for j in range(d):
        h = rng.integers(0, w, size=U)
        g = rng.integers(0, p, size=U)
        ang = (2.0 * np.pi / p) * g
        crR = np.bincount(h[posR], weights=R[1] * np.cos(ang)[posR], minlength=w)
        ciR = np.bincount(h[posR], weights=R[1] * np.sin(ang)[posR], minlength=w)
        crS = np.bincount(h[posS], weights=S[1] * np.cos(ang)[posS], minlength=w)
        ciS = np.bincount(h[posS], weights=S[1] * np.sin(ang)[posS], minlength=w)
        rej[j] = float(np.sum(crR * crS + ciR * ciS))
    return rej

def fixed20(re, im, d=None):
    """Fixed_20 control: always drop the 20% of rows with the largest |Im|."""
    d = d or len(re)
    return float(np.median(re[np.argsort(im)[: int(round(d * 0.8))]]))

def rand20(re, d, seed):
    """Rand_20 control: drop 20% of rows uniformly at random.

    Uses the SAME rng seeding as the archived producer
    (03_tmccs_estimators.eval_config: default_rng(seed + 7_000_000)).
    """
    rng = np.random.default_rng(seed + 7_000_000)
    keep = rng.permutation(d)[: int(round(d * 0.8))]
    return float(np.median(re[keep]))

def fp_estimate(keys, counts=None, p=3, eps=0.1, gam=0.01, seed=0,
                agg="median"):
    """One-call F_p estimate of a stream given as (keys, counts).

    Thin wrapper: builds a ComplexCountSketch sized by (eps, gam) and returns
    its estimate.  Equivalent to cpp_ccs_estimate(keys, counts, p, eps, gam,
    seed) when agg='median'.
    """
    sk = ComplexCountSketch.from_eps(p, eps, gam, seed)
    sk.update(keys, counts if counts is not None else np.ones(len(list(keys))))
    return sk.estimate(agg)

def star_estimate(F, p, width, seed, depth=64, agg="median"):
    """Estimate a p-star (multi-way join size) from k frequency vectors.

    F is a list of k frequency vectors over the SAME key universe: the
    answer is sum_a prod_i f^a_i (the size of the k-way natural join on a
    shared key).  All k columns must be sketched with the SAME h and g --
    that sharing is what makes the element-wise product of counters an
    unbiased estimator (paper Section 5.4).

    Returns the point estimate; `agg` is median or tmccs as above.
    """
    rng = np.random.default_rng(seed)
    re, im = star_rows([np.asarray(f, dtype=np.float64) for f in F],
                       p, width, rng, d=depth)
    return tmccs(re, im) if agg == "tmccs" else median(re)

def join2_estimate(R, S, p, width, seed, depth=64, agg="median"):
    """Estimate a 2-way join size |R join S| with an order-p CCS.

    R and S are (keys, counts) pairs over a shared key universe.  The two
    columns are sketched with matched hashes, and the estimate is
    Re( sum_s C^R_s * conj(C^S_s) ): the exponents +1 and -1 cancel, so one
    stored sketch answers the join at every order (paper Section 5.5).

    Returns the point estimate; `agg` is median or tmccs.
    """
    rng = np.random.default_rng(seed)
    est = ccs_cross2_rows(R, S, p, width, rng, d=depth)
    if agg == "tmccs":
        raise ValueError("tmccs needs |Im|; use ccs_cross2_rows directly")
    return median(est)


# --------------------------------------------------------------------------- #
# Archived C++-pipeline sketch class (Table 2 bit-for-bit)
# --------------------------------------------------------------------------- #

class ComplexCountSketch:
    """An order-p Complex Count Sketch.

    Each of the `depth` rows holds `width` complex counters.  A key a is
    mapped to bucket h(a) and to a p-th root of unity g(a) = omega^{h'(a)},
    omega = exp(2*pi*i/p), using the same 32-bit MurmurHash as the C++
    pipeline:

        bucket = murmur2_32(key, seed + i)             % width
        root   = murmur2_32(key, seed + i*i + 1 + depth) % p

    One row's estimate of F_p = sum_a f_a^p is Re( sum_s C[j, s]^p ); the
    sketch answer is the median over rows (mean of the two middle order
    statistics when `depth` is even), exactly as in the paper.

    Parameters
    ----------
    p : int          -- the order (2, 3, 4, ...).  p = 2 is the classical
                        Count Sketch limit; p >= 3 is where complex roots of
                        unity are needed (see the paper's necessity theorem).
    width : int      -- buckets per row; error ~ width^{-1/2}.
    depth : int      -- rows; failure probability ~ exp(-depth).
    seed : int       -- hash seed (all rows derive their seeds from it).

    Examples
    --------
    >>> import numpy as np, ccs
    >>> rng = np.random.default_rng(0)
    >>> keys = rng.integers(0, 10_000, size=200_000)
    >>> sk = ccs.ComplexCountSketch(p=3, width=1024, depth=8, seed=1)
    >>> sk.update(keys)
    >>> est = sk.estimate()                       # F_3 estimate
    >>> truth = sum(c**3 for c in np.bincount(keys))
    >>> print(f"{est:.4g} vs {truth:.4g}")

    Notes
    -----
    * `update` is additive, so a stream can be fed in chunks and sketches
      built on different machines can be added (`merge`) -- the counters are
      plain sums, which is what makes CCS mergeable.
    * With width, depth = ccs.cpp_dims(p, eps, gam) and the same seed,
      `estimate()` returns exactly `cpp_ccs_estimate(...)`.  Use that form
      when you want the paper's (eps, gam) parameterisation.
    """

    def __init__(self, p, width, depth=64, seed=0):
        if p < 2:
            raise ValueError("p must be at least 2")
        self.p = int(p)
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        self.CR = np.zeros((self.depth, self.width))
        self.CI = np.zeros((self.depth, self.width))

    # -- construction ------------------------------------------------------- #

    def update(self, keys, counts=None):
        """Add `keys` (an iterable of keys) with multiplicities `counts`.

        Keys may be str, bytes or anything str()-able (ints are converted with
        str()).  `counts` defaults to 1 per key.  Returns self, so calls chain.
        """
        keys = list(keys)
        if not keys:
            return self
        if counts is None:
            counts = np.ones(len(keys))
        else:
            counts = np.asarray(counts, dtype=np.float64)
            if len(counts) != len(keys):
                raise ValueError("keys and counts must have the same length")
        kb = [k if isinstance(k, (str, bytes)) else str(k) for k in keys]
        hb, hr = cpp_ccs_hashes(kb, self.seed, self.depth)
        buckets = hb % self.width
        roots = hr % self.p
        ang = (2.0 * math.pi / self.p) * roots
        gr, gi = np.cos(ang), np.sin(ang)
        for i in range(self.depth):
            self.CR[i] += np.bincount(buckets[i], weights=counts * gr[i],
                                      minlength=self.width)
            self.CI[i] += np.bincount(buckets[i], weights=counts * gi[i],
                                      minlength=self.width)
        return self

    @classmethod
    def from_eps(cls, p, eps, gam=0.01, seed=0):
        """Build with the paper's (eps, gam) sizing: width, depth = cpp_dims."""
        width, depth = cpp_dims(p, eps, gam)
        return cls(p, width, depth, seed)

    # -- query -------------------------------------------------------------- #

    def row_values(self):
        """Per-row (re, im): Re and |Im| of sum_s C[j, s]^p.

        C^p is formed by repeated complex multiplication (NOT numpy's complex
        power), which is what the C++ pipeline does and what the analysis
        assumes.
        """
        re = np.empty(self.depth)
        im = np.empty(self.depth)
        for i in range(self.depth):
            cr, ci = self.CR[i], self.CI[i]
            a, b = cr.copy(), ci.copy()
            for _ in range(self.p - 1):
                a, b = a * cr - b * ci, a * ci + b * cr
            re[i] = a.sum()
            im[i] = abs(b.sum())
        return re, im

    def estimate(self, agg="median"):
        """Point estimate; `agg` is one of median / tmccs / mean / med_of_means.

        'tmccs' is the paper's truncated median: rows with large |Im| carry
        most of the error on skewed inputs, so Tukey-truncate on |Im| before
        taking the median (Table 3 and Table 4).
        """
        re, im = self.row_values()
        if agg == "median":
            return median(re)
        if agg == "tmccs":
            return tmccs(re, im)
        if agg == "mean":
            return mean(re)
        if agg == "med_of_means":
            return med_of_means(re)
        raise ValueError(f"unknown aggregator {agg!r}")

    def counters(self):
        """The (CR, CI) counter arrays, shape (depth, width)."""
        return self.CR, self.CI

    def memory_bytes(self):
        """Complex128-equivalent footprint of the counter arrays."""
        return 2 * self.CR.nbytes

    # -- mergeability ------------------------------------------------------- #

    def merge(self, other):
        """Return a new sketch equal to the sum of two sketches.

        Requires identical (p, width, depth, seed): the counters are then
        plain entry-wise sums.  This is the mergeability property -- two
        parties can sketch their own stream and combine without the raw data.
        """
        if (self.p, self.width, self.depth, self.seed) != \
                (other.p, other.width, other.depth, other.seed):
            raise ValueError("sketches are not mergeable (params differ)")
        out = ComplexCountSketch(self.p, self.width, self.depth, self.seed)
        out.CR = self.CR + other.CR
        out.CI = self.CI + other.CI
        return out

    def __add__(self, other):
        return self.merge(other)

    def __repr__(self):
        return (f"ComplexCountSketch(p={self.p}, width={self.width}, "
                f"depth={self.depth}, seed={self.seed})")
