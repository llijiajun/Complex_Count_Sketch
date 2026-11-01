"""Exp 21 -- Sizing a 3-star deployment from the exact two-term variance
(paper Table 6, "Sizing the 3-star at eps=0.1, delta=0.01"), plus the
real-structure infeasibility certificate (Remark rem:real-floor).

Protocol (paper Sec. "Sizing a Star Deployment"):
  * instances : flat uniform x3 / zipf1.5 x3 aligned / mixed u-z1.1-z1.5
                (each column D=10^4 keys, N=2x10^6 draws, independently
                regenerated) and a misaligned instance (D=300 keys/column,
                heavy mass H=20 on disjoint supports, unit shared tail).
  * coefficients (Theorem thm:star-var, exact):
        A = (1/2) sum_{a<b} Mix(a,b)^2
        B = (1/2) sum_{a<b<c} P(a,b,c)^2
    evaluated in time linear in the universe size by expanding Mix and P
    into monomials: every sum reduces to products of
        M_e = sum_x prod_i f^i_x^{e_i},   e in {0,1,2}^3,
    over the 6 proper subsets (A) and the 6x6 bijection pairs (B);
    B further uses inclusion-exclusion over coinciding keys.
  * rule : least w with (A/w + B/w^2) <= eps^2 J^2 / 4, d* = 37 rows;
        pairwise prescription = B:=0, i.e. w = 4 Psi / eps^2 (Psi = A/J^2).
  * pilot : star sketches at widths 128/256/512 with d=2 rows, 3 columns
        = 84 KB charged (matches the KB column of Table 6); (A,B) by
        non-negative least squares on the three row variances, J by the
        row means.
  * certificate : real multipliers carry the width-independent floor
        (mu_6 - 1) Q with Q = sum_x (prod_i f^i_x)^2; feasible iff
        (mu_6-1) Q <= eps^2 J^2 / 4.  Two-point r=sqrt(3/2) has
        mu_6 = 89/8; the golden-ratio multiplier minimises the floor
        factor at mu_6 = 5.
  * metrics : fail = fraction of 40 deployments missing by more than eps;
        med = median relative error.

NOTE on exact numbers: synthetic columns are regenerated here with fixed
seeds, so widths of the same configuration match the paper's Table 6 to
instance-regeneration noise (the pairwise widths, being 4 Psi / eps^2, are
the most sensitive).  The misaligned instance is a faithful reconstruction
(one heavy key of mass H per column on disjoint supports, unit shared
tail); every verdict of the table -- fixed default fails 3 of 4, both
sized rules fail nowhere, the real structure is infeasible on 3 of 4 --
reproduces regardless.

Usage: python exp21_star_sizing.py [deployments_per_row]   # paper: 40
"""
import os, sys
from itertools import combinations, permutations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ccs

EPS, DELTA = 0.1, 0.01
DSTAR = int(np.ceil(8 * np.log(1 / DELTA)))            # 37
N, D = 2_000_000, 10_000
PILOT_WIDTHS = (128, 256, 512)
PILOT_D = 2                                            # 84 KB total charged
PILOT_KB = sum(PILOT_WIDTHS) * PILOT_D * 16 * 3 / 1024
MU6_R = 89.0 / 8.0                                     # two-point sqrt(3/2)
MU6_GOLDEN = 5.0                                       # (phi, -1/phi)


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #

def make_column(name, seed):
    rng = np.random.default_rng(seed)
    if name == "uniform":
        draws = rng.integers(0, D, size=N)
    else:
        alpha = float(name[4:])
        p = np.arange(1, D + 1, dtype=float) ** (-alpha)
        p /= p.sum()
        draws = rng.choice(D, size=N, p=p)
    k, c = np.unique(draws, return_counts=True)
    f = np.zeros(D)
    f[k] = c
    return f


def misaligned(H=20.0, dkeys=300):
    """Heavy mass H on a disjoint support key per column, unit shared tail."""
    f1 = np.ones(dkeys); f1[0] = H
    f2 = np.ones(dkeys); f2[1] = H
    f3 = np.ones(dkeys); f3[2] = H
    return [f1, f2, f3]


# --------------------------------------------------------------------------- #
# Exact A and B, linear in the universe size
# --------------------------------------------------------------------------- #

class MECache:
    """M_e = sum_x prod_i f^i_x^{e_i} for exponent vectors e in {0,1,2}^3."""

    def __init__(self, F):
        self.F = F
        self.cache = {}

    def __call__(self, e):
        e = tuple(int(v) for v in e)
        if e not in self.cache:
            terms = [self.F[i] ** e[i] for i in range(3) if e[i] > 0]
            m = np.ones_like(self.F[0]) if terms else None
            for t in terms:
                m = m * t
            self.cache[e] = float(m.sum()) if terms else float(len(self.F[0]))
        return self.cache[e]


def AB_star(F):
    """Exact A, B, J, Q of Theorem thm:star-var at p=3, arbitrary columns."""
    M = MECache(F)
    X = F[0] * F[1] * F[2]
    J = float(X.sum())
    Q = float((X ** 2).sum())

    # A: Mix = sum over the 6 proper non-empty subsets S of {1,2,3} of
    #    m_S(a,b) = prod_{i in S} f^i_a prod_{i not in S} f^i_b.
    # sum_{a<b} Mix^2 = (1/2) sum_{S,T} [ M_{e(S,T)} M_{2-e(S,T)} - Q ],
    # e_i(S,T) = 1_{i in S} + 1_{i in T} in {0,1,2}.
    subs = []
    for k in (1, 2):
        subs.extend(combinations(range(3), k))
    # sum over ORDERED (S,T) pairs gives sum_{a!=b} Mix^2; A = (1/2) sum_{a<b}
    # = (1/4) sum_{a!=b}.
    A = 0.0
    for S in subs:
        for T in subs:
            e = tuple((i in S) + (i in T) for i in range(3))
            e2 = tuple(2 - v for v in e)
            A += M(e) * M(e2) - Q
    A *= 0.25

    # B: P = sum over the 6 bijections pi of prod_i f^i_{pi(i)}.
    # For a bijection pair (pi, sigma) let v_s in {0,1,2}^3 collect the column
    # exponents assigned to key-slot s in {a,b,c}.  With
    #   S_dt = M_{v_a}M_{v_b}M_{v_c} - M_{v_a+v_b}M_{v_c} - M_{v_a+v_c}M_{v_b}
    #          - M_{v_b+v_c}M_{v_a} + 2 M_{v_a+v_b+v_c}
    # (inclusion-exclusion over coinciding keys), we get
    #   B = (1/12) sum_{pi,sigma} S_dt(pi, sigma).
    bis = list(permutations(range(3)))
    B = 0.0
    for pi in bis:
        for sg in bis:
            v = [np.zeros(3, dtype=int) for _ in range(3)]   # slots a, b, c
            for i in range(3):
                v[pi[i]][i] += 1
                v[sg[i]][i] += 1
            va, vb, vc = (tuple(int(x) for x in u) for u in v)
            vab = tuple(va[j] + vb[j] for j in range(3))
            vac = tuple(va[j] + vc[j] for j in range(3))
            vbc = tuple(vb[j] + vc[j] for j in range(3))
            vabc = tuple(va[j] + vb[j] + vc[j] for j in range(3))
            s = (M(va) * M(vb) * M(vc) - M(vab) * M(vc)
                 - M(vac) * M(vb) - M(vbc) * M(va) + 2 * M(vabc))
            B += s
    B *= 1.0 / 12.0
    return A, B, J, Q


def AB_bruteforce(F):
    """Direct double/triple summation; only feasible for small universes."""
    n = len(F[0])
    f1, f2, f3 = F
    A = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            mix = ((f1[a] + f1[b]) * (f2[a] + f2[b]) * (f3[a] + f3[b])
                   - f1[a] * f2[a] * f3[a] - f1[b] * f2[b] * f3[b])
            A += mix * mix
    A *= 0.5
    # triples a<b<c, vectorised over the strict simplex
    a, b, c = np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                          indexing="ij")
    mask = (a < b) & (b < c)
    ia, ib, ic = a[mask], b[mask], c[mask]
    ga = np.column_stack([f1[ia], f2[ia], f3[ia]])
    gb = np.column_stack([f1[ib], f2[ib], f3[ib]])
    gc = np.column_stack([f1[ic], f2[ic], f3[ic]])
    P = np.zeros(len(ia))
    for pe in permutations(range(3)):
        P += ga[:, pe[0]] * gb[:, pe[1]] * gc[:, pe[2]]
    return A, 0.5 * float((P ** 2).sum())


def nnls_2var(Mmat, v):
    """Tiny exact NNLS for the 2-variable pilot fit (4 sign patterns)."""
    best, bestres = None, np.inf
    for mask in ((1, 1), (0, 1), (1, 0)):
        cols = [Mmat[:, j] for j in (0, 1) if mask[j]]
        if not cols:
            continue
        Aref = np.column_stack(cols)
        sol, *_ = np.linalg.lstsq(Aref, v, rcond=None)
        if np.any(sol < 0):
            continue
        x = np.zeros(2)
        x[[j for j in (0, 1) if mask[j]]] = sol
        res = float(((Mmat @ x - v) ** 2).sum())
        if res < bestres:
            best, bestres = x, res
    return best if best is not None else np.zeros(2)


def width_rule(A, B, J):
    T = EPS ** 2 * J ** 2 / 4
    return int(np.ceil((A + np.sqrt(A * A + 4 * B * T)) / (2 * T)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    PER = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 40
    print(f"=== 3-star sizing (Table 6): eps={EPS}, delta={DELTA}, d*={DSTAR}, "
          f"{PER} deployments per row, pilot {PILOT_KB:.0f} KB charged ===")

    flat = [make_column("uniform", s) for s in (11, 12, 13)]
    zipf = [make_column("zipf1.5", s) for s in (21, 22, 23)]
    mixed = [make_column("uniform", 31), make_column("zipf1.1", 32),
             make_column("zipf1.5", 33)]
    instances = [
        ("flat uniform x3", flat),
        ("zipf1.5 x3 aligned", zipf),
        ("mixed u/z1.1/z1.5", mixed),
        ("misaligned D=300", misaligned()),
    ]

    header = (f"{'instance':20s} {'configuration':16s} {'w':>9s} {'d':>3s} "
              f"{'KB':>9s} {'fail':>6s} {'med':>10s}")
    print(header)
    for name, F in instances:
        A, B, J, Q = AB_star(F)
        # small-universe check of the monomial expansion (both sides on the
        # same truncated universe, D capped at 150)
        nc = min(len(F[0]), 150)
        Fc = [f[:nc] for f in F]
        Ac, Bc, _, _ = AB_star(Fc)
        Ab, Bb = AB_bruteforce(Fc)
        okA = np.isclose(Ac, Ab, rtol=1e-9)
        okB = np.isclose(Bc, Bb, rtol=1e-9)
        print(f"  [check] expansion vs direct summation (D={nc}): "
              f"A {'ok' if okA else 'MISMATCH'}, "
              f"B {'ok' if okB else 'MISMATCH'}")
        w_pair = width_rule(A, 0.0, J)
        w_ideal = width_rule(A, B, J)
        # pilot: three widths, d=2 rows, 3 columns -> (A,B) by NNLS, J by mean
        vs, rows = [], []
        for wd in PILOT_WIDTHS:
            rng = np.random.default_rng(70_000 + wd)
            re, _ = ccs.star_rows(F, 3, wd, rng, d=PILOT_D)
            vs.append(float(np.var(re, ddof=1)))
            rows.extend(re.tolist())
        Jhat = float(np.mean(rows))
        Mmat = np.column_stack([[1 / w for w in PILOT_WIDTHS],
                                [1 / w ** 2 for w in PILOT_WIDTHS]])
        Ap, Bp = nnls_2var(Mmat, np.array(vs))
        w_pilot = width_rule(Ap, Bp, Jhat)
        rows_cfg = [
            ("pairwise rule", w_pair, 6, 0.0, A, 0.0, J),
            ("fixed default", 256, 64, 0.0, A, B, J),
            ("ideal sizing", w_ideal, DSTAR, 0.0, A, B, J),
            ("pilot sizing", w_pilot, DSTAR, PILOT_KB, Ap, Bp, Jhat),
        ]
        # real-structure floor certificate
        for mu6, tag in ((MU6_R, "two-point"), (MU6_GOLDEN, "golden")):
            floor = (mu6 - 1) * Q
            budget = EPS ** 2 * J ** 2 / 4
            print(f"  [floor {tag:6s}] floor/budget = {floor / budget:9.3g} "
                  f"-> {'feasible' if floor <= budget else 'INFEASIBLE'}")
        for cfg, w, d, extra_kb, Au, Bu, Ju in rows_cfg:
            errs = []
            for t in range(PER):
                rng = np.random.default_rng(8000 + 131 * t)
                re, _ = ccs.star_rows(F, 3, w, rng, d=d)
                est = float(np.median(re))
                errs.append(abs(est - J) / J)
            errs = np.array(errs)
            fail = float((errs > EPS).mean())
            med = float(np.median(errs))
            kb = w * d * 16 / 1024 + extra_kb
            print(f"{name:20s} {cfg:16s} {w:9d} {d:3d} {kb:9.1f} "
                  f"{fail:6.3f} {med:10.2e}", flush=True)
    print("EXP21_DONE")
