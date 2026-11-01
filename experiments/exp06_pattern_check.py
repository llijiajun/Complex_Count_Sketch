"""Exp 06 -- exact verification of the p-star variance claims (paper Sec 4.4).

(a) The counterexample  Var[Re R_j] = 3H^2/(2w) + (H^6+3H^4)/(2w^2)  for
    f1=(H,0,0,1), f2=(0,H,0,1), f3=(0,0,H,1), verified EXACTLY by enumerating
    co-location patterns (kernel partitions) with their placement
    probabilities and all 3^4 root assignments per pattern.
(b) The pair-bucket variance 1/2 Mix(a,b)^2 for arbitrary column triples.
(c) The one-item-per-column triple bucket: (A1 A2 A3)^2 / 2.
"""
import itertools, math
import numpy as np


def var_given_placement(FC, assign):
    n = len(FC[0])
    E_R = E_R2 = 0.0
    for g in itertools.product(range(3), repeat=n):
        gr = [np.exp(2j * np.pi * k / 3) for k in g]
        R = 0j
        for s in set(assign):
            Z = 1 + 0j
            for i, f in enumerate(FC):
                y = 0j
                for it in range(n):
                    if assign[it] == s:
                        y += f[it] * gr[it]
                Z = Z * y
            R = R + Z
        E_R += R / 3 ** n
        E_R2 += R.real ** 2 / 3 ** n
    return E_R2 - abs(E_R) ** 2


def partitions(el):
    if not el:
        yield []
    else:
        f, rest = el[0], el[1:]
        for p in partitions(rest):
            for i in range(len(p)):
                yield p[:i] + [[f] + p[i]] + p[i + 1:]
            yield [[f]] + p


if __name__ == "__main__":
    print("=== (a) counterexample, exact pattern enumeration ===")
    for H in (7.0, 50.0):
        FC = [np.array([H, 0, 0, 1.0]), np.array([0, H, 0, 1.0]), np.array([0, 0, H, 1.0])]
        for w in (100, 1000, 10000):
            total = 0.0
            for part in partitions([0, 1, 2, 3]):
                k = len(part)
                assign = {}
                for bi, blk in enumerate(part):
                    for e in blk:
                        assign[e] = bi
                prob = math.prod(range(w - k + 1, w + 1)) / w ** 4
                total += prob * var_given_placement(FC, [assign[e] for e in range(4)])
            pred = 3 * H * H / (2 * w) + (H ** 6 + 3 * H ** 4) / (2 * w ** 2)
            print(f"H={H:g} w={w}: exact={total:.10g} formula={pred:.10g} "
                  f"reldiff={abs(total - pred) / pred:.1e}")

    print("=== (b) pair bucket: Var = Mix^2/2, arbitrary columns ===")
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(20):
        cols = [(rng.uniform(0.5, 8), rng.uniform(0.5, 8)) for _ in range(3)]
        Zs = []
        for ia, ib in itertools.product(range(3), repeat=2):
            ga, gb = np.exp(2j * np.pi * ia / 3), np.exp(2j * np.pi * ib / 3)
            Z = 1 + 0j
            for fa, fb in cols:
                Z = Z * (fa * ga + fb * gb)
            Zs.append(Z)
        Zs = np.array(Zs); re = Zs.real
        v = ((re - re.mean()) ** 2).mean()
        PiA = np.prod([c[0] for c in cols]); PiB = np.prod([c[1] for c in cols])
        M = np.prod([c[0] + c[1] for c in cols]) - PiA - PiB
        bad += abs(v - M * M / 2) > 1e-8 * max(1, v)
    print(f"random column triples: 20, mismatches: {bad}")

    print("=== (c) one-per-column triple: (A1 A2 A3)^2 / 2 ===")
    for A, B, C in ((2.0, 3.0, 5.0), (7.0, 7.0, 7.0)):
        acc = sum((A * B * C * np.exp(2j * np.pi * (ia + ib + ic) / 3)).real ** 2 / 27
                  for ia, ib, ic in itertools.product(range(3), repeat=3))
        print(f"A,B,C={A},{B},{C}: exact={acc:.6f} (ABC)^2/2={(A*B*C)**2/2:.6f}")
    print("EXP-PATTERN DONE")
