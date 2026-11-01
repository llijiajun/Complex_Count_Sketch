"""Exp 16 -- bridge joins via circular convolution (paper Section 5.5).

Star-chain query  A(a) x B(a,b) x C(b)  over synthetic columns:
star contraction (element-wise product of co-keyed sketches) + circular
convolution over C for the bridge table, order 3.  Comparison: the real
per-edge construction at equal width.

Usage: python exp16_bridge_demo.py [W] [d] [trials]

注意: 本脚本是论文 §5.5 桥接段落的紧凑演示实例(自含生成数据)。
论文 §5.5 引用的桥接数字(0.02%–1.1% / 0.04%–1.5%)来自完整归档管线
(results/multijoin_starchain.txt, 由 02_multiway_join.py 生成), 实例不同,
数值不可直接互比。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit()) else 2048
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
D1, D2, N_SIDE, N_BR = 64, 64, 20_000, 4_000
P = 3


def zipf_keys(D, a, n, rng):
    if a is None:
        return rng.integers(0, D, size=n)
    out, got = [], 0
    while got < n:
        k = rng.zipf(a, size=2 * n) - 1
        k = k[k < D]
        out.append(k); got += len(k)
    return np.concatenate(out)[:n]


def gen(dist1, dist2, seed):
    rng = np.random.default_rng(seed)
    fA = np.bincount(zipf_keys(D1, dist1, N_SIDE, rng), minlength=D1).astype(float)
    fC = np.bincount(zipf_keys(D2, dist2, N_SIDE, rng), minlength=D2).astype(float)
    a = zipf_keys(D1, dist1, N_BR, rng)
    b = zipf_keys(D2, dist2, N_BR, rng)
    fB = np.zeros((D1, D2))
    np.add.at(fB, (a, b), 1.0)
    return fA, fC, fB


def conv(x, y):
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))


def run(fA, fB, fC, W, d, seed, real=False):
    pairs = np.argwhere(fB > 0)
    pa, pb, pf = pairs[:, 0], pairs[:, 1], fB[fB > 0]
    # prime-stride projection shared by ALL tables, so that the bridge index
    # equals the convolution index: posA(x) + posC(y) = posB(x,y) (mod W)
    posA = (101 * np.arange(D1)) % W
    posC = (7 * np.arange(D2)) % W
    hB = (101 * pa + 7 * pb) % W
    est = np.empty(d)
    rng = np.random.default_rng(seed)
    for j in range(d):
        if real:
            sAB = rng.integers(0, 2, size=len(pa)) * 2 - 1
            sBC = rng.integers(0, 2, size=len(pb)) * 2 - 1
            sA = np.zeros(D1); np.add.at(sA, pa, sAB)          # A: edge AB
            sC = np.zeros(D2); np.add.at(sC, pb, sBC)          # C: edge BC
            sB = sAB * sBC                                     # B: both edges
            CA = np.bincount(posA, weights=fA * sA, minlength=W)
            CB = np.bincount(hB, weights=pf * sB, minlength=W)
            CC = np.bincount(posC, weights=fC * sC, minlength=W)
            MB = CB * conv(CA, CC)
            est[j] = float(MB.sum())
        else:
            # order-3, per-key exponents (1,2): key a: A:1, B:2; key b: B:2, C:1
            gA = np.exp(2j * np.pi * rng.integers(0, P, size=D1) / P)
            gC = np.exp(2j * np.pi * rng.integers(0, P, size=D2) / P)
            gBa = gA[pa] ** 2                                  # B shares key a's root
            gBb = gC[pb] ** 2                                  # B shares key b's root
            CA = np.zeros(W, dtype=complex); np.add.at(CA, posA, fA * gA)
            CB = np.zeros(W, dtype=complex); np.add.at(CB, hB, pf * gBa * gBb)
            CC = np.zeros(W, dtype=complex); np.add.at(CC, posC, fC * gC)
            MB = CB * conv(CA, CC)
            est[j] = float(np.real(MB.sum()))
    return est


if __name__ == "__main__":
    errs_ccs, errs_real = [], []
    for t in range(TRIALS):
        fA, fC, fB = gen(1.1, 1.5, seed=100 + t)
        truth = float((fA[:, None] * fB * fC[None, :]).sum())
        e1 = run(fA, fB, fC, W, DEPTH, seed=1000 + t, real=False)
        e2 = run(fA, fB, fC, W, DEPTH, seed=2000 + t, real=True)
        errs_ccs.append(abs(np.median(e1) - truth) / truth)
        errs_real.append(abs(np.median(e2) - truth) / truth)
    print(f"star-chain A(a)xB(a,b)xC(b), W={W}, d={DEPTH}, {TRIALS} trials")
    print(f"CCS order-3: mean relative error = {np.mean(errs_ccs):.4g}")
    print(f"per-edge real: mean relative error = {np.mean(errs_real):.4g}")
    print("EXP-BRIDGE DONE")
