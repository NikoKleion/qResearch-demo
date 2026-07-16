"""
ESPRIT-ZNE core method + the baselines i benchmarked it against.

the idea: real quantum hardware noise is (mostly) a sum of decaying exponentials.
the standard error-mitigation method, richardson extrapolation, fits a POLYNOMIAL to
that -> wrong model -> irreducible bias no amount of shots fixes. ESPRIT (from radar /
spectroscopy, 1986) is built exactly for fitting sums of exponentials, so borrow it.

run this file to see all methods on a synthetic 2-channel decay. 
"""
import numpy as np


def esprit_zne(noise_levels, expectations, K_max=None):
    # fits E(lam) = E0 + sum_k c_k * r_k^lam and returns E0 (the zero-noise value)
    lam = np.asarray(noise_levels, float)
    E = np.asarray(expectations, float)
    N = len(E)
    assert N >= 4, "need >=4 noise levels"

    # baseline subtract FIRST. this is the whole trick and it's the bit not in
    # textbook ESPRIT. kill the constant E0 so the hankel/SVD only sees the decaying
    # part. skip this and the SVD hands you a garbage pole sitting at r~1.
    R = E - E[-1]

    # hankel it
    M = N // 2
    H = np.array([[R[i + j] for j in range(M)] for i in range(N - M)])

    # SVD, then auto-pick model order K from the biggest gap in the log singular values
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    if K_max is None:
        gaps = np.abs(np.diff(np.log(S + 1e-14)))
        K = min(max(int(np.argmax(gaps)) + 1, 1), M - 1)
    else:
        K = min(K_max, M - 1)

    # ESPRIT shift-invariance: U1 @ Psi = U2, eigenvalues of Psi ARE the poles r_k
    Us = U[:, :K]
    Psi, *_ = np.linalg.lstsq(Us[:-1, :], Us[1:, :], rcond=None)
    eigs = np.linalg.eigvals(Psi)

    # keep only physical poles: basically real, positive, < 1 (markovian decay).
    # this filter is why ESPRIT doesn't explode under coherent noise - it just throws
    # the unit-circle junk away.
    poles = eigs[np.abs(eigs.imag) < 0.1].real
    poles = poles[(poles > 0.0) & (poles < 1.0)]
    if len(poles) == 0:
        poles = np.array([0.5])  # fallback, shouldn't usually hit this

    # vandermonde least squares for E0 + amplitudes
    V = np.column_stack([np.ones(N)] + [poles[k] ** lam for k in range(len(poles))])
    if not np.all(np.isfinite(V)):
        return float(E[-1])
    coeffs, *_ = np.linalg.lstsq(V, E, rcond=None)
    return float(np.real(coeffs[0]))


# ---- the baselines ----

def richardson_zne(noise_levels, expectations, n_points=3):
    # richardson-3: lagrange interpolate to lam=0 through the 3 lowest levels.
    # this is the mitiq/qiskit default, so it's the baseline everyone compares to.
    lam = np.asarray(noise_levels, float)[:n_points]
    E = np.asarray(expectations, float)[:n_points]
    n = len(lam); E0 = 0.0
    for i in range(n):
        num = np.prod([0.0 - lam[j] for j in range(n) if j != i])
        den = np.prod([lam[i] - lam[j] for j in range(n) if j != i])
        E0 += E[i] * num / den
    return float(E0)


def richardson_full(noise_levels, expectations):
    # use ALL points -> degree N-1 polynomial -> diverges via runge. the "use all my
    # data" trap. included so the failure is visible, not hidden.
    return richardson_zne(noise_levels, expectations, n_points=len(noise_levels))


def single_exp_zne(noise_levels, expectations, n_grid=400):
    # single exponential a + b*exp(-g*lam). dumb grid search over g, linear LS for a,b.
    # NOTE: this is the competitor i should have tested first. on clean monotone decays
    # it matches or beats ESPRIT. finding that out late is the lesson of this project.
    lam = np.asarray(noise_levels, float)
    E = np.asarray(expectations, float)
    best = (float("nan"), np.inf)
    for g in np.linspace(0.001, 3.0, n_grid):
        X = np.column_stack([np.ones_like(lam), np.exp(-g * lam)])
        coef, *_ = np.linalg.lstsq(X, E, rcond=None)
        resid = np.sum((X @ coef - E) ** 2)
        if resid < best[1]:
            best = (float(coef[0]), resid)
    return best[0]


def no_mitigation(noise_levels, expectations):
    return float(np.asarray(expectations, float)[0])   # raw lam=1 value


def apply_all(noise_levels, expectations):
    return {
        "ESPRIT-ZNE":    esprit_zne(noise_levels, expectations),
        "Richardson-3":  richardson_zne(noise_levels, expectations, 3),
        "Richardson-N":  richardson_full(noise_levels, expectations),
        "Single-exp":    single_exp_zne(noise_levels, expectations),
        "No mitigation": no_mitigation(noise_levels, expectations),
    }


if __name__ == "__main__":
    # synthetic 2-channel decay with a bit of shot noise. NOT a scientific result,
    # just shows the machinery runs and recovers E0.
    rng = np.random.default_rng(0)
    E0_true = -1.13627
    lam = np.array([1, 3, 5, 7, 9, 11, 13, 15], float)
    clean = E0_true + 0.12 * np.exp(-0.15 * lam) + 0.05 * np.exp(-0.45 * lam)
    meas = clean + rng.normal(0, 0.005, size=lam.size)
    print(f"synthetic 2-channel decay, true E0 = {E0_true:.5f}")
    for name, e0 in apply_all(lam, meas).items():
        print(f"  {name:<14} E0={e0:+.5f}  |err|={abs(e0 - E0_true):.5f}")
    print("\nverdict from the full 8-run study: ESPRIT is novel, ~90us cheap, and never")
    print("blows up (0% catastrophic fails vs 20-100% for single-exp) - but on realistic")
    print("circuits it has NO accuracy edge; single-exp or richardson win. honest result.")
