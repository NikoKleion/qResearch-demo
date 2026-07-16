"""
LDP-Cert: the rarity sweep (the test that settled it).

problem: useful fault-tolerant machines need logical error rates ~1e-12. nobody can measure
that - brute-force monte carlo would need ~1e12 shots to see one failure. so borrow rare-event
statistics (large deviations / importance sampling, the stuff insurance + physics use for
"estimate a probability too rare to ever observe"). key choice: put the ACTUAL decoder in the
loop, so the rare event is "decoder answers wrong", not a cheap error-weight proxy.

testbed: d=5, T=3 space-time repetition code with exact-ML decoding. exact_PL enumerates the
truth so every point is validated. tilted IS = our method; naive MC + p-extrapolation = the
things it competes with.
"""
import numpy as np
from itertools import product


def state_bits(d):
    s = np.arange(2 ** d)
    return ((s[:, None] >> np.arange(d)) & 1).astype(np.int8)


def apply_transition(alpha, d, p):
    # push the accumulated-error distribution through one round of bit-flip channels
    N = alpha.shape[0]
    for j in range(d):
        a = alpha.reshape(N, 2 ** (d - 1 - j), 2, 2 ** j)
        s = a[:, :, 0, :] + a[:, :, 1, :]
        a[:, :, 0, :] *= (1 - 2 * p); a[:, :, 0, :] += p * s
        a[:, :, 1, :] = s - a[:, :, 0, :]
    return alpha


def run_is(d, T, p, q, N, pt, qt, seed, chunk=15000, collect_moments=False):
    # tilted importance sampling. sample errors at the (bigger) proposal rates pt,qt so
    # failures actually happen, then reweight by the true/proposal likelihood ratio.
    tot_v = tot_v2 = 0.0; tot_fail = 0; wf_all = []
    mom = np.zeros(2); mom_w = 0.0; done = 0
    while done < N:
        n = min(chunk, N - done)
        r = np.random.default_rng(seed * 100003 + done)
        nS = 2 ** d; bits = state_bits(d)
        synb = (bits[:, :-1] ^ bits[:, 1:]).astype(np.float32)
        syn_abs = synb.sum(1); cls = (bits.sum(1) > d // 2)
        lq, l1q = np.log(q), np.log(1 - q)
        lrp = np.log(p / pt) - np.log((1 - p) / (1 - pt)); basep = d * np.log((1 - p) / (1 - pt))
        lrq = np.log(q / qt) - np.log((1 - q) / (1 - qt)); baseq = (d - 1) * np.log((1 - q) / (1 - qt))
        alpha = np.zeros((n, nS), np.float32); alpha[:, 0] = 1.0
        e = np.zeros((n, d), np.int8); logw = np.zeros(n)
        fsum = np.zeros(n); msum = np.zeros(n)
        for t in range(T):
            f = (r.random((n, d)) < pt).astype(np.int8)
            m = (r.random((n, d - 1)) < qt).astype(np.int8)
            logw += f.sum(1) * lrp + basep + m.sum(1) * lrq + baseq
            fsum += f.sum(1); msum += m.sum(1); e ^= f
            obs = ((e[:, :-1] ^ e[:, 1:]) ^ m).astype(np.float32)
            alpha = apply_transition(alpha, d, p)
            ham = obs.sum(1)[:, None] + syn_abs[None, :] - 2.0 * (obs @ synb.T)
            alpha *= np.exp(ham * (lq - l1q) + (d - 1) * l1q)
            alpha /= alpha.sum(1, keepdims=True)
        dec = alpha[:, cls].sum(1) > 0.5
        fail = dec != (e.sum(1) > d // 2)
        w = np.exp(logw); vals = fail * w
        tot_v += vals.sum(); tot_v2 += (vals ** 2).sum(); tot_fail += int(fail.sum())
        wf_all.append(w[fail])
        if collect_moments and fail.any():
            ww = w[fail]
            mom += np.array([(ww * fsum[fail]).sum() / (T * d), (ww * msum[fail]).sum() / (T * (d - 1))])
            mom_w += ww.sum()
        done += n
    mean = tot_v / N
    wf = np.concatenate(wf_all) if wf_all else np.array([])
    ess = (wf.sum() ** 2 / (wf ** 2).sum()) if len(wf) else 0.0
    out = [mean, ess, tot_fail]
    if collect_moments:
        out.append((mom / mom_w) if mom_w > 0 else np.array([p, q]))
    return out


def naive_mc(d, T, p, q, N, seed):
    return run_is(d, T, p, q, N, p, q, seed)      # proposal == truth, so weights = 1


def exact_PL(d, T, p, q):
    # exact decoder-inclusive P_L by enumerating every observation sequence. this is the
    # ground truth nothing here is allowed to disagree with.
    nS = 2 ** d; bits = state_bits(d)
    synb = bits[:, :-1] ^ bits[:, 1:]; cls = bits.sum(1) > d // 2
    nO = 2 ** (d - 1)
    obits = ((np.arange(nO)[:, None] >> np.arange(d - 1)) & 1).astype(np.int8)
    ham = (obits[:, None, :] ^ synb[None, :, :]).sum(2)
    Em = q ** ham * (1 - q) ** (d - 1 - ham)
    PL = 0.0
    for obs_seq in product(range(nO), repeat=T):
        alpha = np.zeros((1, nS)); alpha[0, 0] = 1.0
        for o in obs_seq:
            alpha = apply_transition(alpha.copy(), d, p) * Em[o]
        m1 = alpha[0, cls].sum()
        PL += min(alpha[0].sum() - m1, m1)
    return PL


def ce_tilt(d, T, p, q, budget, seed, iters=3, per=2500, start=(0.08, 0.06)):
    # cross-entropy: harvest where failures actually happen, aim the tilt there, repeat.
    # if too few failures, just push the tilt harder. THIS is the deployable method.
    pt, qt = start; used = 0
    for it in range(iters):
        res = run_is(d, T, p, q, per, pt, qt, seed + it, collect_moments=True)
        used += per
        if res[2] >= 10:
            pt = float(np.clip(res[3][0], p, 0.45)); qt = float(np.clip(res[3][1], q, 0.45))
        else:
            pt, qt = min(pt * 1.6, 0.4), min(qt * 1.6, 0.4)
    return run_is(d, T, p, q, max(budget - used, 3000), pt, qt, seed + 50)[0]


def p_extrapolation(d, T, p_target, q, budget, seed, anchors=(0.10, 0.13, 0.16, 0.20)):
    # current practice: measure where failures are common (high p), fit a power law, and
    # extend it down to the rare target. cheap. also usually wrong in the tail.
    per = budget // len(anchors); xs, ys = [], []
    for i, pa in enumerate(anchors):
        m = naive_mc(d, T, pa, q, per, seed + i)[0]
        if m > 0: xs.append(np.log(pa)); ys.append(np.log(m))
    if len(xs) < 2: return np.nan
    b, a = np.polyfit(xs, ys, 1)
    return float(np.exp(a + b * np.log(p_target)))


if __name__ == "__main__":
    D, T, BUD, R = 5, 3, 12000, 3
    pgrid = [0.05, 0.02, 0.008, 0.003, 0.001]     # common -> deep tail
    print(f"d={D} T={T}, exact truth every point, matched budget {BUD}, R={R}")
    print(f"{'P_L(truth)':>11} | {'naive relRMSE':>13} | {'CE-tilt (ours)':>14} | {'p-extrap':>10}")
    for p in pgrid:
        truth = exact_PL(D, T, p, p)
        def rmse(fn):
            es = np.array([fn(s) for s in range(R)])
            es = es[np.isfinite(es)]
            return np.sqrt(np.mean((es - truth) ** 2)) / truth if len(es) else np.nan
        rn = rmse(lambda s: naive_mc(D, T, p, p, BUD, 100 + s)[0])
        rc = rmse(lambda s: ce_tilt(D, T, p, p, BUD, 200 + s))
        rp = rmse(lambda s: p_extrapolation(D, T, p, p, BUD, 300 + s))
        print(f"{truth:11.2e} | {rn:13.2f} | {rc:14.2f} | {rp:10.2f}")
    print("\nread the columns: common events (top) everyone's fine, naive MC is even simplest.")
    print("go down into the tail and naive MC dies (0 failures -> relRMSE 1.0) and p-extrap")
    print("stays badly biased, while CE-tilt keeps low error. BUT up top it buys nothing - so")
    print("this is a deep-tail specialist, not a general tool. honest on both ends.")
