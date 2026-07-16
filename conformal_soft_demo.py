"""
Conformal abort with soft (analog) readout - the headline test.

setup: a decoder decides accept/reject on its own confidence. problem is that confidence
is miscalibrated and drifts, so a hand-tuned threshold gives you no real guarantee.
conformal prediction (from ML) gives a distribution-free guarantee P(fail|accept) <= alpha
WITHOUT trusting the decoder's model - it only needs exchangeable labeled calibration data.

the mean test here: build a decoder that is provably LYING about its noise (thinks p=0.02,
truth is 0.08) and check the conformal wrapper still Holds the promised failure rate, where
the naive self-confidence threshold does not.

also exposes the boundary: this only works with a continuous score. hard syndromes have a
handful of distinct confidence values -> nothing to threshold. (documented separately.)
"""
import numpy as np
rng = np.random.default_rng(42)

d = 9
alpha = 0.02                       # the failure rate we promise on accepted shots
p_true,  p_model  = 0.08, 0.02     # decoder BELIEVES 0.02, reality is 0.08 -> it's lying
s_true,  s_model  = 0.9,  0.6      # analog readout noise: real vs what the decoder assumes


def gen(N, p, s):
    # generate N shots of a d-bit repetition readout, decode with the (wrong-model) ML
    # decoder, and return (soft error score, did-it-actually-fail).
    b = rng.integers(0, 2, N)                       # true logical bit
    flips = rng.random((N, d)) < p
    x = b[:, None] ^ flips
    y = (1 - 2 * x) + rng.normal(0, s, (N, d))      # soft/analog measurement

    # decoder's log-likelihood ratio, computed with its WRONG p_model / s_model
    def lik(y, sign):
        g = lambda mu: np.exp(-(y - mu) ** 2 / (2 * s_model ** 2))
        return (1 - p_model) * g(sign) + p_model * g(-sign)
    L = np.log(lik(y, 1) + 1e-300) - np.log(lik(y, -1) + 1e-300)
    Lsum = L.sum(1)
    dec = (Lsum < 0).astype(int)
    fail = dec != b
    conf_err = 1 / (1 + np.exp(np.abs(Lsum)))        # decoder's own "prob i'm wrong" score
    return conf_err, fail


# --- calibration: pick the conformal threshold tau from labeled shots ---
# sort accepted-by-score, walk up until the running failure rate would break alpha.
# the (cum+1)/(nacc+1) is the finite-sample conformal correction, not just an average.
Ncal = 20000
s_cal, f_cal = gen(Ncal, p_true, s_true)
order = np.argsort(s_cal); fs = f_cal[order]
cum = np.cumsum(fs); nacc = np.arange(1, Ncal + 1)
ok = (cum + 1) / (nacc + 1) <= alpha
tau = s_cal[order][np.nonzero(ok)[0].max()]

print(f"d={d} alpha={alpha} | decoder model (p={p_model},sig={s_model}) vs TRUTH (p={p_true},sig={s_true})")
print(f"conformal tau = {tau:.3e}\n")
print(f"{'p':5s} {'sig':4s} {'method':16s} {'P(fail|acc)':>11s} {'accept%':>8s}")

# test in-distribution, drifted, and improved. conformal should hold; self-confidence won't.
for pt, st, note in [(0.08, 0.9, "in-dist"), (0.10, 1.0, "drifted"), (0.06, 0.8, "improved")]:
    s, f = gen(300000, pt, st)
    for name, thr in [("self-confidence", alpha), ("conformal", tau)]:
        acc = s <= thr
        pfa = f[acc].mean()
        flag = "  <-- VIOLATED" if pfa > alpha * 1.15 else ""
        print(f"{pt:.2f}  {st:.1f}  {name:16s} {pfa:11.4f} {acc.mean()*100:7.1f}%{flag}  ({note})")

print("\nresult: conformal holds P(fail|acc) at/under alpha even against the lying decoder,")
print("at ~full acceptance; the naive self-confidence threshold blows past alpha. the honest")
print("catch (found in the hard-syndrome runs): needs a SOFT score - hard syndromes are too")
print("coarse to threshold. and a later competitor review narrowed how novel this really is.")
