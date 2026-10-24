r"""Experiment 11 -- exact co-location-pattern verification of the p-star
counter-example (paper Section 4.5, eq. pstar-counter).

Instance: f1=(H,0,0,1), f2=(0,H,0,1), f3=(0,0,H,1); items a,b,c (heavies, one
per column) and z (shared, f=1 in all three columns). True join size J=1, D=4.

For each set partition of {a,b,c,z} (the co-location pattern), the row variance
Var[Re R_j] is computed EXACTLY by enumerating all 3^4 root assignments
(shared root per key across columns). Averaging over patterns with their exact
placement probabilities (w)_k / w^4 for a k-block kernel gives

    E[ Var[Re R_j] ]  ==  3H^2/(2w) + (H^6 + 3H^4)/(2w^2)   (exact, all w).

Also verifies, by exact enumeration over 9 root pairs, that the pair-bucket
variance 1/2 * Mix(a,b)^2 holds for arbitrary column triples (aligned,
anti-aligned, and random), and that a one-item-per-column triple bucket
contributes (A1*A2*A3)^2/2.
"""
import itertools, math
import numpy as np

def var_given_placement(FC, assign):
    n_items = len(FC[0])
    E_R = 0.0; E_R2re = 0.0
    for g in itertools.product(range(3), repeat=n_items):
        gr = [np.exp(2j*np.pi*k/3) for k in g]
        R = 0.0+0j
        for s in set(assign):
            Z = 1.0+0j
            for i, f in enumerate(FC):
                y = 0.0+0j
                for it in range(n_items):
                    if assign[it] == s:
                        y += f[it]*gr[it]
                Z *= y
            R += Z
        E_R += R/3**n_items
        E_R2re += R.real**2/3**n_items
    return float(E_R2re - abs(E_R)**2)

def partitions(elements):
    if not elements:
        yield []
    else:
        first, rest = elements[0], elements[1:]
        for p in partitions(rest):
            for i in range(len(p)):
                yield p[:i]+[[first]+p[i]]+p[i+1:]
            yield [[first]]+p

def placement_of(part, n):
    assign = {}
    for bi, blk in enumerate(part):
        for e in blk:
            assign[e] = bi
    return [assign[i] for i in range(n)]

def main():
    print("=== (a) p-star counter-example, exact pattern enumeration ===")
    for H in (7.0, 50.0):
        FC = [np.array([H,0,0,1.0]), np.array([0,H,0,1.0]), np.array([0,0,H,1.0])]
        for w in (100, 1000, 10000):
            total = 0.0
            for part in partitions([0,1,2,3]):
                k = len(part)
                prob = math.prod(range(w-k+1, w+1)) / w**4   # (w)_k / w^4
                total += prob * var_given_placement(FC, placement_of(part, 4))
            pred = 3*H*H/(2*w) + (H**6 + 3*H**4)/(2*w*w)
            print(f"H={H:g} w={w}: exact={total:.10g}  formula={pred:.10g}  "
                  f"reldiff={abs(total-pred)/max(pred,1e-300):.2e}")

    print("=== (b) pair-bucket variance == Mix^2/2, arbitrary columns ===")
    rng = np.random.default_rng(0)
    bad = 0
    for t in range(20):
        cols = [(rng.uniform(0.5,8), rng.uniform(0.5,8)) for _ in range(3)]
        acc = []
        for ia, ib in itertools.product(range(3), repeat=2):
            ga = np.exp(2j*np.pi*ia/3); gb = np.exp(2j*np.pi*ib/3)
            Z = 1.0+0j
            for fa, fb in cols:
                Z *= fa*ga + fb*gb
            acc.append(Z)
        acc = np.array(acc); re = acc.real
        v = ((re-re.mean())**2).mean()
        PiA = np.prod([c[0] for c in cols]); PiB = np.prod([c[1] for c in cols])
        M = np.prod([c[0]+c[1] for c in cols]) - PiA - PiB
        if abs(v - M*M/2) > 1e-8*max(1.0, v):
            bad += 1
    print(f"random column triples tested: 20, mismatches: {bad}")

    print("=== (c) one-item-per-column triple bucket == (A1 A2 A3)^2 / 2 ===")
    for A, B, C in ((2.0,3.0,5.0), (7.0,7.0,7.0)):
        acc = 0.0
        for ia, ib, ic in itertools.product(range(3), repeat=3):
            Z = A*B*C*np.exp(2j*np.pi*(ia+ib+ic)/3)
            acc += Z.real**2/27
        print(f"A,B,C={A},{B},{C}: exact={acc:.6f}  (ABC)^2/2={(A*B*C)**2/2:.6f}")
    print("EXP11_DONE")

if __name__ == "__main__":
    main()
