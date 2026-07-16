# Research Demo

This folder is a short walk through how I actually work, using the single most
conclusive test from each of four projects. I'm not dumping every run here (yet), just the
one experiment per project that settled the question, plus the reasoning that got me to
it and what I concluded. The code for each is in this folder and runs on its own.

The four projects look unrelated on the surface — error mitigation, code structure, rare
failure statistics, decoder decisions. The point of this demo is to show what I have learned.


WORK IN PROGRESS: THIS IS CURRENTLY BEING BUILT OFF OF MY RAW AND PERSONAL FILES

## The four conclusive tests, and how they connect

### 1 — ESPRIT-ZNE  
Borrowed the ESPRIT algorithm from radar/spectroscopy to do zero-noise extrapolation,
because real hardware noise is a sum of exponentials and the standard method (Richardson)
fits a polynomial to it — wrong model. The method is real: novel, fast, and it doesn't blow
up where the naive baselines do.

The lesson this project taught me, the hard way, is step 3. I benchmarked against
Richardson for a long time before realizing the *real* competitor was plain
single-exponential fitting, and once I tested against it honestly, ESPRIT had no accuracy
advantage on realistic circuits. So the conclusion is honest and specific: novel + reliable,
but not more accurate, and here's exactly which method wins where. I did more thorough competition testing ever since.

## The Idea

Every quantum computer running today is noisy. One of the most practical fixes for that noise, called zero-noise extrapolation (ZNE), works by running your circuit at several amplified noise levels and fitting a curve backward to where the noise would be zero. The problem is that everyone fits a polynomial to data that physically cannot be polynomial. Quantum hardware noise is a sum of decaying exponentials, not polynomial terms, and fitting the wrong model introduces a bias that no number of additional measurements can fix. A 1986 radar signal processing algorithm called ESPRIT was built to solve the exponential fitting problem exactly, using a linear algebra approach that has been numerically stable for forty years. This project applies ESPRIT to quantum ZNE for the first time.

---

## Where the Math Comes From

In 1795, Gaspard de Prony discovered that data following sums of decaying exponentials could be fit by solving a linear recurrence relation, no nonlinear optimization needed. Mathematically correct, numerically catastrophic under noise. The roots of the characteristic polynomial would scatter wildly when data was imperfect, making it useful on paper and useless in practice.

In 1986, Roy, Paulraj, and Kailath at Stanford published ESPRIT for radar direction-finding. The problem was identical to Prony's: given noisy measurements of a sum of complex exponentials, recover the frequencies. ESPRIT solved it by forming a Hankel matrix from the data, decomposing it via SVD to isolate the signal subspace, then exploiting a shift-invariance property of that subspace to extract the exponential parameters. Instead of rooting a high-degree polynomial, ESPRIT solves a small eigenvalue problem. Numerically stable, information-theoretically near-optimal. It became a standard tool in radar, spectroscopy, and MRI.

The connection to quantum ZNE is direct. Under Markovian noise, the dominant noise model for superconducting qubits, the expectation value of any observable follows:

```
E(λ) = E₀ + Σ_{k=1}^{K}  c_k · exp(−γ_k · λ)
```

where λ is the noise amplification factor, E₀ is the zero-noise value we want, γ_k are the Lindbladian eigenvalues, and r_k = exp(−γ_k) are the noise poles. This is exactly the Prony model. The math ESPRIT was built for in 1986 is exactly the math quantum noise follows in 2026.

Richardson extrapolation fits a polynomial to this. That is the wrong model. Paper arxiv:2502.20673 now has formal theorems proving exactly how wrong: explicit bias and variance bounds showing the approximation error is irreducible, and that Richardson's extrapolation coefficients grow exponentially with polynomial degree. The same paper proves that using more noise levels with Richardson makes things worse, not better. The benchmarks confirm this: Richardson-N at N=8 gives 14.3x worse accuracy than ESPRIT, and at N=14 the mean error reaches 24, physically impossible for a bounded observable.

---

## The Algorithm

**Step 1: Baseline subtraction.** Subtract the noisiest measurement: R(λ) = E(λ) − E(λ_max). This removes the E₀ baseline so the Hankel matrix sees only the exponential components. Without this step, the algorithm finds a spurious pole at r ≈ 1 and fails. This step is not in classical ESPRIT. It was derived fresh for this application and is the most critical implementation detail.

**Step 2: Hankel matrix.** From N baseline-subtracted measurements, build the (N−M) × M matrix H where H[i,j] = R[i+j] and M = N//2. The true rank of this matrix equals K, the number of independent decoherence channels.

**Step 3: SVD and model order.** Decompose H = U·Σ·Vᵀ. Singular values drop sharply at rank K. Estimate K automatically from the largest gap in log(σ_i).

**Step 4: ESPRIT shift invariance.** Take the K leading left singular vectors. Partition into U₁ (all rows except last) and U₂ (all rows except first). The shift-invariance property guarantees U₁·Ψ = U₂, where eigenvalues of Ψ are exactly the noise poles {r_k}. Solve for Ψ via least squares, take eigenvalues. Filter to physical poles: real, positive, less than 1.

**Step 5: Vandermonde least squares.** With poles known, build V[:,0] = 1 and V[:,k] = r_k^λ. Solve V·[E₀, c₁, ..., c_K]ᵀ = E(λ) in least squares. The first coefficient is E₀.



### 2 — Matroid-Erasure  (`matroid_exchange_test.py`)
A pure structural question: for classical codes, the erasure-correctable sets form a
*matroid* (that's the theorem behind greedy decoding). Does the quantum version hold? Instead
of arguing, I just computed it over real stabilizer codes. The
answer is no, with hand-checkable counterexamples, and the reason is quantum degeneracy.

This is the cleanest win of the four finished projects precisely because it's exact (no sampling, no model
assumptions, no decoder) and the question was crisp. Different lesson from ESPRIT: when you
can make a question exact, do there's nothing to argue with. A clean hypothesis and data matters the most.

## The Idea

When a qubit is lost (a photon leaves the cavity, an atom escapes the tweezer,
or a dual-rail qubit heralds an erasure), the decoder knows WHERE the error is,
just not what it was. Erasure is the cleanest noise in quantum computing and
the leading noise model for several hardware platforms. For classical codes,
150 years of combinatorics answers every erasure question through one object:
the code's matroid. Which erasure patterns are recoverable, whether greedy
peeling decoders are optimal, closed-form thresholds: all matroid facts.

Nobody had asked whether the quantum analogue holds. We asked, computed the
answer exhaustively on real codes, and found: no. The failure is not a
technicality; it is caused by degeneracy, the defining quantum feature of
quantum codes. But the failure is also not total: rare stabilizer codes DO
carry nontrivial matroid structure, and nobody knows which or why.

## Where the Math Comes From

Whitney (1935) abstracted linear independence into the matroid axioms; the
exchange axiom is what makes greedy algorithms provably optimal (Rado 1957,
Edmonds 1971). Greene (1976) tied code weight enumerators to the Tutte
polynomial. Erasure decoding of a classical [n,k] code succeeds iff the erased
positions are independent in the code's matroid (textbook material).

On the quantum side, an erasure set E is correctable iff no nontrivial logical
operator is supported inside E (the Knill–Laflamme condition specialized to
located errors). This is a symplectic GF(2) rank condition, computable
exactly, which is what makes this project's every claim checkable.

## The problem

Characterize the stabilizer codes whose erasure family is a matroid.
Data so far: uniform families (MDS-like and 'erasure-tight' codes) always;
a sparse population of non-uniform matroidal codes (explicit [[5,1,2]],
[[6,1,2]] examples in run_3); everything else fails via degeneracy.
Candidate route: the family is matroidal iff the degenerate-bonus sets
themselves form a compatible circuit structure; first test: identify the
run_3 [[6,1,2]] matroid (graphic? transversal?) and reverse-engineer.



### 3 — LDP-Cert  (`ldp_rarity_sweep.py`)
Useful quantum computers need logical error rates around 1e-12 that no experiment can
measure. Insurance and physics have dealt with "estimate a probability too rare to ever
observe" for decades (large-deviation theory, importance sampling), so I brought that in and
put the decoder in the loop.

Here I applied the ESPRIT lesson deliberately: I swept the whole rarity axis and benchmarked
against every competitor including current practice, all validated against exact truth. The
result is honest on both sides, it's the only method left standing deep in the tail where
everything else dies or silently lies, and it buys nothing in the common regime where a
plain count is fine. Knowing exactly where a method stops helping is the point.


## The Idea

Useful fault-tolerant machines need logical error rates near 1e-12, roughly one
failure per month of continuous running per logical qubit. Nobody can measure
that. Today's substitute is to measure where failures are common and
extrapolate downward, with no guarantee, exactly where the expensive
architecture decisions live. Insurance mathematicians solved this problem
shape decades ago: they price once-a-century floods from thirty years of
records using large deviations theory, which gives the exact exponential decay
rate of rare events, identifies the most probable way disaster happens (the
optimal fluctuation, or instanton), and licenses a simulation trick: make
failure common on purpose, in precisely the right way, then mathematically
undo the bias. This project imports that machinery into quantum error
correction with one defining choice: the rare event is "the decoder answers wrong,"
an algorithm's output rather than a
closed-form error set. That is what makes the instanton nontrivial: it must
fool the decoder, not merely be large. This connection had not been computed.

## Where the Math Comes From

Cramér (1938) computed exact exponential rates for tails of i.i.d. sums; our
code-capacity warm-up is literally his theorem, which is why it makes a
perfect validation anchor. Varadhan (1966) and Freidlin-Wentzell (1970s)
extended the theory to path spaces, where rare events happen along the
minimizer of a rate functional, which field theory calls the instanton. Siegmund (1976)
proved that tilting the sampling distribution toward that minimizer is the
optimal change of measure, and, in the same stroke, that tilting toward the
wrong point blows the variance up; both halves of that theorem appear as
measurements in this repo. The effective-sample-size diagnostic is from
sequential Monte Carlo (Kong-Liu-Wong 1994); the cross-entropy tilt search is
Rubinstein's method with per-species Bernoulli moments. On the QEC side the
space-time decoding picture is Dennis-Kitaev-Landahl-Preskill (2002), whose
stat-mech mapping gives thresholds; our object is the sub-threshold exponent
with the decoder inside the event, a first-passage-percolation quantity
(Kesten) that the two literatures never connected.

### 4 — Conformal-Abort  (`conformal_soft_demo.py`)
Decoders make accept/reject calls on confidence signals that are miscalibrated and drift.
Conformal prediction (from ML) gives distribution-free guarantees without trusting the model,
so I wrapped it around the decoder's abort decision. The conclusive test builds a decoder that
is provably lying about its own confidence and shows the wrapper still delivers the promised
failure rate , with one catch it also exposed: it needs a continuous (soft) score, hard
syndromes are too coarse.

## The Idea

Every QEC decoder produces a confidence signal that is unreliable: matching-weight
gaps, BP marginals, neural softmax scores, all miscalibrated and drifting
with the device. Yet fault-tolerant protocols increasingly make accept/reject
decisions on exactly these signals: magic-state cultivation postselects,
adaptive schemes abort risky shots, hierarchical decoders escalate hard cases.
Today those thresholds are hand-tuned numbers with no guarantee attached.

Conformal prediction, a 20-year-old statistics framework now standard in
high-stakes ML, converts any uncalibrated score into decisions with
finite-sample, distribution-free guarantees, using only exchangeability of
calibration data. This project is the first (to our searches) to wrap it
around a quantum decoder's abort decision, and the first to show the adaptive
variant surviving realistic noise drift with delayed labels.

## Where the Math Comes From

Vovk, Gammerman & Shafer (2005) built conformal prediction on one idea: if
calibration and test data are exchangeable, the rank of a new nonconformity
score among calibration scores is uniform, so quantiles transfer with exact
finite-sample coverage, no model assumptions. Lei et al. popularized the
cheap split-conformal form. Angelopoulos et al. (2022) generalized coverage
to arbitrary monotone risks (conformal risk control, which is what the abort rule
uses). Gibbs & Candès (2021) made it adaptive: a control-loop update of the
working level tracks distribution shift at the cost of a sliding, windowed
guarantee. QEC is an unusually good host: shots are naturally batched,
labels (logical outcomes) genuinely arrive, and miscalibration + drift are
the defining pain of real devices.


## what each file actually reproduces (the numbers that mattered)

every script is real math / simulation, runs on its own (numpy only), and prints the exact
data the project turned on. two per project — the conclusive test plus the finding behind it.

**ESPRIT-ZNE**
- `esprit_zne_method.py` — the algorithm itself + baselines; smoke test where single-exp beats it.
- `esprit_reliability.py` — MC over 600 random decays/noise: ESPRIT ~0% catastrophic vs
  single-exp climbing to ~22% and Richardson-N to ~74%. the real pitch (won't blow up), reproduced.

**Matroid-Erasure**  (exact, exhaustive, hand-checkable — no sampling)
- `matroid_exchange_test.py` — [[5,1,3]] matroidal; Steane 84 exchange violations; surface 532.
- `matroid_submodularity.py` — the polymatroid rescue also fails: Steane 378 submodularity
  violations, surface 1768. MDS/perfect codes stay clean.

**LDP-Cert**  (validated against exact enumeration)
- `ldp_rarity_sweep.py` — the crossover: CE-tilt is the only method holding low error into the
  deep tail; naive MC dies, p-extrapolation stays biased; buys nothing in the common regime.
- `ldp_silent_bias.py` — why naive IS is dangerous: two single-knob tilts disagree ~40% and
  are confidently wrong vs exact truth; the joint tilt nails it.

**Conformal-Abort**
- `conformal_soft_demo.py` — holds P(fail|accept) ≤ α around a decoder that's provably lying,
  at full acceptance, where the naive threshold doesn't.
- `conformal_drift_aci.py` — static cert VIOLATES under drift (0.040 overall, 0.086 last
  quarter); the adaptive fix holds every quarter, paying with a lower accept rate. no retuning.


  ## this demo is a work in progress. figures, real tests, and more thorough notes from my actual files are on the way soon. 
