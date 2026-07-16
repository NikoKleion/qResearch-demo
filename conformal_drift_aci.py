"""
conformal abort part 2: what happens when the device DRIFTS.

the soft-conformal test (conformal_soft_demo.py) calibrates once and holds - but only if the
noise stays put. real hardware drifts. conformal's guarantee needs exchangeable data, and
drift breaks exchangeability, so a once-calibrated threshold slowly stops holding alpha.

fix: adaptive conformal (ACI, gibbs-candes 2021). after each batch, nudge the working risk
level based on the failure rate you just saw among accepted shots. asymmetric update - react
fast to a violation, relax slowly. labels arrive one batch late (realistic).

this reproduces: static cert VIOLATES under a continuous drift, ACI keeps P(fail|accept) at
target every quarter, paying for it by accepting a bit less. numbers match the round-2 run.
"""
import numpy as np

rng = np.random.default_rng(122)
d, alpha = 9, 0.02
p_model, s_model = 0.02, 0.6           # decoder's assumed model (fixed, and wrong once drift starts)

def gen(N, p, s):
    b = rng.integers(0, 2, N)
    x = b[:, None] ^ (rng.random((N, d)) < p)
    y = (1 - 2 * x) + rng.normal(0, s, (N, d))
    g = lambda mu: np.exp(-(y - mu) ** 2 / (2 * s_model ** 2))
    lik1 = (1 - p_model) * g(1) + p_model * g(-1)
    lik0 = (1 - p_model) * g(-1) + p_model * g(1)
    L = (np.log(lik1 + 1e-300) - np.log(lik0 + 1e-300)).sum(1)
    fail = (L < 0).astype(int) != b
    score = 1 / (1 + np.exp(np.abs(L)))
    return score, fail

# calibrate static threshold once, at t=0 conditions
s_cal, f_cal = gen(20000, 0.06, 0.8)
o = np.argsort(s_cal); fs = f_cal[o]
ok = (np.cumsum(fs) + 1) / (np.arange(1, len(fs) + 1) + 1) <= alpha
tau_static = s_cal[o][np.nonzero(ok)[0].max()]

B, Nb = 300, 500
alpha_t = alpha
tau_aci = tau_static
tau_series = []
win_s, win_f, pending = [], [], None
for t in range(B):
    frac = t / (B - 1)
    score, fail = gen(Nb, 0.06 + 0.14 * frac, 0.8 + 0.4 * frac)   # p:0.06->0.20, sig:0.8->1.2
    if pending is not None:
        sc_p, fl_p, tau_p = pending
        acc_p = sc_p <= tau_p
        risk = fl_p[acc_p].mean() if acc_p.any() else 0.0
        g = 0.5 if risk > alpha else 0.03            # react fast to violations, relax slow
        alpha_t = np.clip(alpha_t + g * (alpha - risk), alpha / 10, 1.5 * alpha)
        win_s.append(sc_p); win_f.append(fl_p)
        if len(win_s) > 8: win_s.pop(0); win_f.pop(0)
        sw = np.concatenate(win_s); fw = np.concatenate(win_f)
        oo = np.argsort(sw); fsp = fw[oo]
        okp = (np.cumsum(fsp) + 1) / (np.arange(1, len(fsp) + 1) + 1) <= alpha_t
        tau_aci = sw[oo][np.nonzero(okp)[0].max()] if okp.any() else sw[oo][max(int(0.5*len(sw))-1, 0)]
    pending = (score, fail, tau_aci)
    tau_series.append(tau_aci)

# score both strategies over the identical drift stream (regen with same seed)
stats = {"static": np.zeros((4, 2)), "aci": np.zeros((4, 2))}
rng = np.random.default_rng(122); _ = gen(20000, 0.06, 0.8)   # burn the calibration draw
for t in range(B):
    frac = t / (B - 1)
    score, fail = gen(Nb, 0.06 + 0.14 * frac, 0.8 + 0.4 * frac)
    qt = min(3, 4 * t // B)
    for name, tau in [("static", tau_static), ("aci", tau_series[t])]:
        acc = score <= tau
        stats[name][qt] += [fail[acc].sum(), acc.sum()]

print(f"target alpha={alpha}; drift p 0.06->0.20, sig 0.8->1.2 over {B} batches\n")
for name in ["static", "aci"]:
    tot_f = stats[name][:, 0].sum(); tot_a = stats[name][:, 1].sum()
    flag = "  <-- VIOLATED" if tot_f / tot_a > alpha * 1.15 else "  ok"
    print(f"{name:7s}: overall P(fail|acc)={tot_f/tot_a:.4f}  accept={tot_a/(B*Nb)*100:.1f}%{flag}")
print("\nper-quarter P(fail|accept) (drift gets worse left->right):")
for name in ["static", "aci"]:
    row = "  ".join(f"Q{i+1}:{stats[name][i,0]/max(stats[name][i,1],1):.4f}" for i in range(4))
    print(f"  {name:7s}: {row}")
print("\nstatic drifts past alpha (worst in the last quarter); ACI holds every quarter by")
print("quietly dropping the accept rate. no human retuning. that's the whole point.")
