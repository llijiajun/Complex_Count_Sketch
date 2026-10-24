r"""Experiment 13 -- isolating support misalignment from skew (paper Section
5.4). Three columns over D=2000 keys; the frequency MULTISET of every column
is the same truncated-Zipf(alpha) vector (marginals fixed exactly); column 1
assigns it in order, columns 2 and 3 keep it aligned except that each of the
top-40 heavy masses moves, independently with probability L, onto a uniformly
random light key.  L = 0 is the aligned deployment, L = 1 fully displaced
heavies.  Because only heavy-to-light placement changes, any error growth in L
is misalignment, not skew.

Reports mean relative error of the CCS p-star estimator (plain median,
w=256, d=64, 60 families) and the exact pairwise misalignment measure
Psi = sum_{a<b} Mix(a,b)^2 / (2 J^2) computed from the true frequencies, for
alpha in {0.8, 1.5} and L in {0, .25, .5, .75, 1}.
"""
import numpy as np

D, N, H = 2000, 2_000_000, 40
W, DROWS, P, T = 256, 64, 3, 60


def make_cols(alpha, L, rng):
    w = np.arange(1, D + 1, dtype=np.float64) ** (-alpha)
    f = (w / w.sum() * N).round()
    heavy = np.arange(H)
    light = np.arange(H, D)
    cols = []
    base = f.copy()
    cols.append(base)
    for _ in range(2):
        g = f.copy()
        move = rng.random(H) < L
        dst = rng.choice(light, size=int(move.sum()))
        np.add.at(g, dst, g[heavy[move]])
        g[heavy[move]] = 0.0
        cols.append(g)
    return cols


def truth_and_psi(cols):
    prod = cols[0] * cols[1] * cols[2]
    J = float(prod.sum())
    # Mix(a,b) over all pairs: Pi(fa+fb) - Pi(fa) - Pi(fb), vectorized
    fa = np.stack(cols)                      # 3 x D
    pa = fa.prod(axis=0)                     # D
    PiSum = (fa[:, None, :] + fa[:, :, None]).prod(axis=0)   # D x D
    Mix = PiSum - np.outer(pa, np.ones(D)) - np.outer(np.ones(D), pa)
    np.fill_diagonal(Mix, 0.0)               # the paper's sum runs over a < b
    psi = float((Mix ** 2).sum() / 2 / (2 * J ** 2))
    return J, psi


def run(cols, seed0):
    U = D
    J = float((cols[0] * cols[1] * cols[2]).sum())
    E = np.empty(T)
    for t in range(T):
        rng = np.random.default_rng(seed0 + t)
        re = np.empty(DROWS)
        for j in range(DROWS):
            h = rng.integers(0, W, size=U)
            g = rng.integers(0, P, size=U)
            ang = (2.0 * np.pi / P) * g
            gr, gi = np.cos(ang), np.sin(ang)
            pr = np.ones(W); pi = np.zeros(W)
            for f in cols:
                cr = np.bincount(h, weights=f * gr, minlength=W)
                ci = np.bincount(h, weights=f * gi, minlength=W)
                pr, pi = pr * cr - pi * ci, pr * ci + pi * cr
            re[j] = pr.sum()
        E[t] = abs(float(np.median(re)) - J) / J
    return E.mean()


if __name__ == "__main__":
    print(f"# alpha, L, mean rel err, Psi (= sum Mix^2/(2J^2)), J")
    for alpha in (0.8, 1.5):
        for L in (0.0, 0.25, 0.5, 0.75, 1.0):
            rng = np.random.default_rng(424242)
            cols = make_cols(alpha, L, rng)
            J, psi = truth_and_psi(cols)
            err = run(cols, seed0=1000 + int(100 * alpha + 10 * L))
            print(f"alpha={alpha:.1f} L={L:.2f} err={err:.4g} psi={psi:.4g} J={J:.6g}",
                  flush=True)
    print("EXP13_DONE", flush=True)
