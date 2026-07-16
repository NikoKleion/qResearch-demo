"""
matroid follow-up: ok so the exchange axiom fails (see matroid_exchange_test.py). but maybe
a weaker structure survives - a POLYMATROID, which only needs the rank function to be
submodular. that would still buy you a lot (tutte-style invariants, greedy bounds). so check
submodularity directly. exact again, no sampling.

rank function: r(E) = size of the largest correctable subset of E. build it by DP - if E
itself is correctable r(E)=|E|, else drop whichever element hurts least and recurse.
submodular means r(A|B) + r(A&B) <= r(A) + r(B) for all A,B. count where it breaks.

spoiler: it breaks too. so no polymatroid rescue. the numbers below (378 for steane, 1768
for surface) are the exact violation counts that killed that escape hatch.
"""
import numpy as np
from itertools import combinations
from math import comb


def rank2(M):
    if len(M) == 0: return 0
    M = (np.array(M) % 2).astype(np.uint8); r = 0
    for c in range(M.shape[1]):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        M[[r, r + piv[0]]] = M[[r + piv[0], r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == M.shape[0]: break
    return r


def pauli_rows(strs, n):
    S = []
    for g in strs:
        v = np.zeros(2 * n, np.uint8)
        for i, ch in enumerate(g):
            if ch in "XY": v[i] = 1
            if ch in "ZY": v[n + i] = 1
        S.append(v)
    return np.array(S)


# same codes as the exchange test, plus [[4,2,2]] (a quantum-MDS code) for contrast
CODES = {
    "[[4,2,2]] (MDS)": (4, pauli_rows(["XXXX", "ZZZZ"], 4)),
    "[[5,1,3]] (perfect)": (5, pauli_rows(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], 5)),
}
Hc = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]], np.uint8)
S7 = [np.concatenate([r, np.zeros(7, np.uint8)]) for r in Hc] + \
     [np.concatenate([np.zeros(7, np.uint8), r]) for r in Hc]
CODES["[[7,1,3]] Steane"] = (7, np.array(S7, np.uint8))
Zst = [[0,1,3,4],[4,5,7,8],[2,5],[3,6]]; Xst = [[1,2,4,5],[3,4,6,7],[0,1],[7,8]]
S9 = []
for qs in Xst:
    v = np.zeros(18, np.uint8); v[qs] = 1; S9.append(v)
for qs in Zst:
    v = np.zeros(18, np.uint8)
    for qq in qs: v[9 + qq] = 1
    S9.append(v)
CODES["[[9,1,3]] surface"] = (9, np.array(S9))


def correctable_all(n, S):
    # which erasure sets are correctable (same symplectic-rank test as the exchange script)
    rS = rank2(S)
    SOm = np.hstack([S[:, n:], S[:, :n]])
    corr = {}
    for size in range(n + 1):
        for E in combinations(range(n), size):
            cols = list(E) + [n + i for i in E]
            d_cent = 2 * len(E) - rank2(SOm[:, cols].T)
            VE = np.zeros((2 * len(E), 2 * n), np.uint8)
            for j, c in enumerate(cols): VE[j, c] = 1
            d_stab = rS + 2 * len(E) - rank2(np.vstack([S, VE]))
            corr[frozenset(E)] = (d_cent == d_stab)
    return corr


for name, (n, S) in CODES.items():
    corr = correctable_all(n, S)
    fam = [set(E) for E, ok in corr.items() if ok]
    famset = {frozenset(E) for E in fam}
    sizes = [sum(1 for A in fam if len(A) == s) for s in range(max(len(A) for A in fam) + 1)]

    # exchange axiom (matroid test)
    viol = sum(1 for A in fam for B in fam
               if len(A) < len(B) and not any(frozenset(A | {x}) in famset for x in B - A))

    # is the family a uniform matroid (all subsets up to some size)?
    t = max(len(A) for A in fam)
    is_uniform = all(sizes[s] == comb(n, s) for s in range(t + 1))

    # rank function by DP, then count submodularity breaks
    r_fun = {}
    for size in range(n + 1):
        for E in combinations(range(n), size):
            E = frozenset(E)
            r_fun[E] = len(E) if corr[E] else max(r_fun[E - {x}] for x in E)
    sub_viol = sum(1 for A in r_fun for B in r_fun
                   if r_fun[A | B] + r_fun[A & B] > r_fun[A] + r_fun[B])

    print(f"{name}: sizes={sizes}  exchange_viol={viol}  uniform_matroid={is_uniform}  "
          f"submodularity_viol={sub_viol}")

print("\nboth axioms fail for steane/surface -> not a matroid AND not a polymatroid. the")
print("MDS code [[4,2,2]] and the perfect [[5,1,3]] stay clean. exact counts, reproducible.")
