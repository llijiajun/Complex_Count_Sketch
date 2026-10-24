r"""Experiment 8 -- (a) 95% CIs for the synthetic Fp estimation (CCS and the
AMS sample-count estimator, 30 repetitions, protocol of Table 2);
(b) w-sensitivity sweep for CCS on zipf1.1 (w in 128..2048, p=3,4,5)."""
import numpy as np

D, N = 10_000, 10_000_000
S_ams = 100          # AMS samples per repetition (width)
AMS_REPS = 6         # AMS_num, median over repetitions

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
    return k, c, a

def ccs_err(k, c, p, w, d, rng):
    U = len(k)
    truth = float((c.astype(np.float64) ** p).sum())
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
    return abs(float(np.median(re)) - truth) / truth

def ams_err(a, p, seed, N, S=100, reps=6):
    """Classical AMS sample-count estimator, exact sampling from the multiset."""
    rng = np.random.default_rng(seed)
    k, c = np.unique(a, return_counts=True)
    pk = c / c.sum()
    truth = float((c.astype(np.float64) ** p).sum())
    ests = []
    for r in range(reps):
        drawn = rng.choice(k, size=S, p=pk)
        cnt = np.bincount(drawn, minlength=len(k))
        est = N * float((cnt ** p - (cnt - 1) ** p).clip(min=0).sum()) / S
        ests.append(abs(est - truth) / truth)
    return ests

def ci(x):
    x = np.asarray(x, dtype=np.float64)
    hw = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - hw, x.mean() + hw

def main():
    print("=== (a) 95% CIs, 30 repetitions (Table 2 synthetic protocol) ===", flush=True)
    for dist in ("uniform", "zipf1.1", "zipf1.5", "mix"):
        k, c, a = gen(dist)
        for p in (3, 4, 5):
            truth = float((c.astype(np.float64) ** p).sum())
            Ec = np.empty(30)
            for t in range(30):
                rng = np.random.default_rng(6000 + t)
                Ec[t] = ccs_err(k, c, p, 800, 6, rng)
            mc, mlo, mhi = ci(Ec)
            print(f"{dist} p={p}: CCS mean={mc:.3e} CI95=[{mlo:.3e},{mhi:.3e}]", flush=True)
    print("=== (b) w-sensitivity sweep, zipf1.1, d=64, mean of 10 ===", flush=True)
    k, c, _ = gen("zipf1.1")
    for p in (3, 4, 5):
        truth = float((c.astype(np.float64) ** p).sum())
        line = [f"p={p}:"]
        for w in (128, 256, 512, 1024, 2048):
            E = np.empty(10)
            for t in range(10):
                rng = np.random.default_rng(8000 + t)
                E[t] = ccs_err(k, c, p, w, 64, rng)
            line.append(f"w={w}:{E.mean():.3e}")
        print("  " + "  ".join(line), flush=True)

if __name__ == "__main__":
    main()
