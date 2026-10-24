r"""Experiment 6 -- Ganguly (RANDOM 2004) estimator vs CCS at equal space.

Ganguly's estimator (his Sec 3.1 / Thm 4): B buckets; each of s1 independent
copies keeps Z_b = sum_{keys in b} f_key * omega^{e_key} (omega = primitive
k-th root, e uniform per key); the estimate is mean_{b,copies} Re(Z_b^k).
Equal space: B*s1 = 65536 complex counters = CCS (w=1024, d=64).
His prescribed copy count s1 = 6*2^k*k^{3k}/eps^2 (Thm 4, eps=0.1) implies a
space 198x CCS's at k=3 -- reported as a factor, and run at 2x CCS space.
"""
import numpy as np

D, N = 10_000, 10_000_000
K = 3

def gen(dist, seed=7):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        a = rng.integers(0, D, size=N)
    elif dist == "mix":
        a1 = rng.integers(0, D, size=N // 2)
        wts = np.arange(1, D + 1) ** (-1.1)
        a2 = rng.choice(D, size=N - N // 2, p=wts / wts.sum())
        a = np.concatenate([a1, a2])
    else:
        wts = np.arange(1, D + 1) ** (-float(dist[4:]))
        a = rng.choice(D, size=N, p=wts / wts.sum())
    k, c = np.unique(a, return_counts=True)
    return k, c

def ganguly(k, c, truth, k_order, B, s1, seed0):
    U = len(k)
    errs = np.empty(10)
    for t in range(10):
        rng = np.random.default_rng(seed0 + t)
        est = 0.0
        for j in range(s1):
            e = rng.integers(0, k_order, size=U)
            ang = (2.0 * np.pi / k_order) * e
            hb = rng.integers(0, B, size=U)
            zr = np.bincount(hb, weights=c * np.cos(ang), minlength=B)
            zi = np.bincount(hb, weights=c * np.sin(ang), minlength=B)
            pr, pi = zr.copy(), zi.copy()
            for _ in range(k_order - 1):
                pr, pi = pr * zr - pi * zi, pr * zi + pi * zr
            est += pr.sum()
        est /= s1
        errs[t] = abs(est - truth) / truth
    return errs

def ccs_fp(k, c, p, w, d, seed0):
    U = len(k)
    truth = float((c.astype(np.float64) ** p).sum())
    E = np.empty(10)
    for t in range(10):
        rng = np.random.default_rng(seed0 + t)
        re = np.empty(d)
        for j in range(d):
            h = rng.integers(0, w, size=U)
            g = rng.integers(0, p, size=U)
            ang = (2.0 * np.pi / p) * g
            cr = np.bincount(h, weights=c * np.cos(ang), minlength=w)
            ci = np.bincount(h, weights=c * np.sin(ang), minlength=w)
            pr, pi = 1.0 + 0 * cr, 0.0 * cr
            for _ in range(p):
                pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
            re[j] = pr.sum()
        E[t] = abs(float(np.median(re)) - truth) / truth
    return E

def main():
    B = 100                                  # Ganguly's prescribed B = n^{1-1/(k-1)}
    s1_equal = 65536 // B                    # equal space to CCS w=1024 d=64
    s1_presc = int(6 * (2**K) * (K**(3*K)) / 0.1**2)   # his Thm-4 prescription, eps=0.1
    print(f"Ganguly configs: B={B}  equal-space s1={s1_equal}  "
          f"prescribed s1={s1_presc} (space {B*s1_presc/65536:.0f}x CCS)", flush=True)
    for dist in ("zipf1.1", "zipf1.5", "uniform", "mix"):
        k, c = gen(dist)
        truth = float((c.astype(np.float64) ** K).sum())
        g_eq = ganguly(k, c, truth, K, B, s1_equal, 7000)
        ccs = ccs_fp(k, c, K, 1024, 64, 8000)
        print(f"{dist} k={K}: CCS(w=1024,d=64) mean={ccs.mean():.3e} med={np.median(ccs):.3e} | "
              f"Ganguly equal-space mean={g_eq.mean():.3e} med={np.median(g_eq):.3e}", flush=True)

if __name__ == "__main__":
    main()
