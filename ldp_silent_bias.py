"""
LDP-Cert, the scary finding: naive importance sampling LIES with a straight face.

the whole method rests on tilting the noise so failures show up, then reweighting. the
obvious thing is to tilt only the data-error rate (one knob). in a space-time code that's
wrong - the instanton also needs measurement errors - so a data-only tilt misses part of the
failure region and comes back BIASED, but with normal-looking error bars. you'd never know
unless you had ground truth.

this reproduces it: at a rare point where we KNOW the exact answer, two different single-knob
tilts disagree with each other and with truth, while the joint (data+meas) tilt nails it.
that gap is why the real method tilts both knobs and cross-checks.

reuses the exact engine from ldp_rarity_sweep.py so it's the same math, not a re-derivation.
"""
import numpy as np
from ldp_rarity_sweep import exact_PL, run_is, naive_mc

D, T = 5, 3
p = q = 0.003                     # rare-ish: exact truth ~5e-6
N = 60000                         # big-ish so what we see is BIAS, not just noise
truth = exact_PL(D, T, p, q)
print(f"d={D} T={T} p=q={p}   EXACT truth P_L = {truth:.4e}\n")
print(f"{'method':30s} {'estimate':>11s} {'error vs truth':>15s}")

def show(label, est):
    print(f"{label:30s} {est:11.3e} {(est-truth)/truth*100:+14.1f}%")

# naive MC - probably barely any failures this deep
show("naive MC", naive_mc(D, T, p, q, N, 1)[0])

# single-knob tilts: crank ONLY the data rate, leave measurement at truth. two versions.
show("single-tilt (data x4)",  run_is(D, T, p, q, N, min(4*p,0.4), q, 11)[0])
show("single-tilt (data x8)",  run_is(D, T, p, q, N, min(8*p,0.4), q, 12)[0])

# joint tilt: push BOTH knobs into the failure region. this is what the real method does.
show("joint tilt (data+meas)", run_is(D, T, p, q, N, 0.09, 0.05, 13)[0])

a = run_is(D, T, p, q, N, min(4*p,0.4), q, 11)[0]
b = run_is(D, T, p, q, N, min(8*p,0.4), q, 12)[0]
print(f"\nthe two single-knob tilts disagree by {abs(a-b)/min(a,b)*100:.0f}% - and both are")
print("off from truth, with innocent error bars. joint tilt matches. that's the silent bias:")
print("a wrong tilt doesn't fail loudly, it hands you a confident wrong number. caught only")
print("because we had exact truth to check against.")
