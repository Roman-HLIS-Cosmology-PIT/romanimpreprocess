import math
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike
from scipy.special import betaln, gammainc, gammaincc, loggamma, logsumexp
from scipy.stats import beta as sp_beta
from scipy.stats import betaprime, gamma, invgamma, t

# TO-DO:

# 1. Batch draws for Type 4 Devroye samlper (>10x quicker) NOPE NO IT'S NOT
# 2. Discuss making asymmetric tolerances for masks, make type4 ==> type5
# 3. Pearson type 5 possible sign error in skewness
# 4. Code error analysis use analytical moments instead of MC analysis
# 5. Analyze errors


def draw_from_Pearson(
    tilnu_21: float, tilnu_31: float, tilnu_41: float, I: np.ndarray, *, atol: float = 0.0, rng=None
):
    """
    Add a Pearson-family random deviate to each element of *I*,
    chosen according to its β₁ & β₂.

    Parameters
    ----------
    tilnu_21, tilnu_31, tilnu_41 : float
        Scalar central-moment ratios (𝜈̃₂₁, 𝜈̃₃₁, 𝜈̃₄₁).
    I : ndarray
        Input intensities.  They are **modified in place**.
    atol : float, optional
        Equality tolerance for the region boundaries.

    Returns
    -------
    draws : random draws from the Pearson family
        with desired moments from eq x in Overleaf
    """
    if rng is None or not hasattr(rng, "random"):
        rng = np.random.default_rng(rng)

    I_clipped = np.clip(np.asarray(I, dtype=float), 3.0, None)

    tilnu_42 = 3 * tilnu_21**2

    # Pearson variables
    b_1 = tilnu_31**2 / (tilnu_21**3 * I_clipped)  # β₁
    b_2 = (tilnu_42 * I_clipped + tilnu_41) / (tilnu_21**2 * I_clipped)  # β₂

    # ---  Pearson admissability  ----------------------------------

    # Base admissibility region (elements outside *base* stay unchanged.)
    base = (b_2 > 0) & (b_1 >= 0) & (b_2 > b_1 + 1) & (b_2 > 0.75 * b_1)

    if not np.any(base):
        """print('-----------------------------')
        print('Returning zeroes')
        print('tilnu21 =', tilnu_21)
        print('tilnu31 =', tilnu_31)
        print('tilnu41 =', tilnu_41)
        print('I_arr (clipped) =', I_clipped)
        print('-----------------------------')"""
        return np.zeros_like(I_clipped, dtype=float)
    # Useful variables for masks
    lhs = b_2
    rhs1 = 1.5 * b_1 + 3
    rhs2 = (48 + 39 * b_1 + 6 * (4 + b_1) ** 1.5) / (32 - b_1)
    # rhs3 = 1.875 * b_1 + 4.5      # 15/8 b₁ + 36/8

    # Debugging
    # print('b_2: ',b_2)
    # print('Type 3: ',rhs1)
    # print('Type 6: greater than ',rhs1,' and less than ',rhs2)
    # print('Type 5: ',rhs2)
    # print('Type 4: greater than ',rhs2,' and less than ',rhs3) NOPE
    # print('Type 4: greater than ',rhs2)

    # Equality bands (exclusive)
    eq1 = np.isclose(lhs, rhs1, atol=atol, rtol=0)  # → Type 3
    eq2 = np.isclose(lhs, rhs2, atol=atol, rtol=0)  # → Type 5

    # Strict zones shrunk by ±atol
    lt1 = lhs < rhs1 - atol
    gt1 = lhs > rhs1 + atol
    lt2 = lhs < rhs2 - atol
    gt2 = lhs > rhs2 + atol
    # lt3 = lhs <  rhs3 - atol

    # Masks: mutually exclusive
    type1 = base & lt1
    type3 = base & eq1
    type5 = base & eq2
    type6 = base & gt1 & lt2
    type4 = base & gt2 & (b_1 < 32)  # & lt3

    # TEST, REMOVE LATER
    overlap = (
        type1.astype(int) + type3.astype(int) + type5.astype(int) + type6.astype(int) + type4.astype(int)
    ) > 1

    if np.any(overlap):
        """print('ERROR:')
        print('Overlap=')
        print('t1',type1.astype(int))
        print('t3',type3.astype(int))
        print('t4',type4.astype(int))
        print('t5',type5.astype(int))
        print('t6',type6.astype(int))
        print('tilnu_21: ',tilnu_21)
        print('tilnu_31: ',tilnu_31)
        print('tilnu_41: ',tilnu_41)
        print('I_arr =', I)
        print('I_clipped =', I_clipped)"""
        raise RuntimeError("Overlapping Pearson masks.")

    # ---  Draws and in-place addition  ----------------------------------
    draws = np.zeros_like(I_clipped, dtype=float)

    if np.any(type1):
        draws[type1] = random_from_type1(tilnu_21, tilnu_31, tilnu_41, I_clipped[type1], rng=rng)
    if np.any(type3):
        draws[type3] = random_from_type3(tilnu_21, tilnu_31, I_clipped[type3], rng=rng)
    if np.any(type5):
        draws[type5] = random_from_type5(tilnu_21, tilnu_31, I_clipped[type5], rng=rng)
    if np.any(type6):
        draws[type6] = random_from_type6(tilnu_21, tilnu_31, tilnu_41, I_clipped[type6], rng=rng)
    if np.any(type4):
        draws[type4] = random_from_type4(tilnu_21, tilnu_31, tilnu_41, I_clipped[type4], rng=rng)

    return draws


# ============================================================
# Helper function to scale and shift a distribution
# ============================================================


def _to_x(y, scale, shift, sign):
    """
    Transforms our distribution:

    1. Scales our distribution to have the correct variance
    2. Translates our distribution to draw the correct mean (zero mean)
    3. Changes the sign of the draw to obtain the correct skew

    Notice: Scale only transforms "y". Does not transform "shift".
    """
    return sign * (scale * y - shift)


# ============================================================
# ============================================================


# Pearson Type 1 (aka Beta) distribution


# ============================================================
# ============================================================


# Helpers for closed-form solution
# ============================================================
def analytic_u_v_from_betas(beta1, beta2):
    """
    Solve for u=a+b and v=(a-b)^2/(ab) from:
      beta1 = 4 v (u+1)/(u+2)^2,
      beta2 = 3 + 6 [ v(u+1) - (u+2) ] / [(u+2)(u+3)].
    """
    u_denom = (beta2 - 3) - 1.5 * beta1
    # if np.any(np.isclose(u_denom, 0.0)):
    #    raise ValueError("Degenerate analytic case: denominator≈0 for at least one element")
    u = 3 * (beta1 - beta2 + 1) / u_denom
    v = beta1 * (u + 2) ** 2 / (4 * (u + 1))
    return u, v


def ab_from_u_v(u, v, want_negative_skew):
    """
    Solves for shape parameters a (alpha) and b (beta):
        u = a + b
        v = (a-b)^2 / (ab)

    There are two solutions, (a_+,b_+), (a_-,b_-),
    which generate opposite skew distributions.
    """
    s = np.sqrt(v / (v + 4))
    a_plus = 0.5 * u * (1 + s)
    b_plus = 0.5 * u * (1 - s)

    if np.isscalar(u):  # scalar shortcut
        if want_negative_skew:
            return (a_plus, b_plus) if a_plus > b_plus else (b_plus, a_plus)
        else:
            return (a_plus, b_plus) if a_plus < b_plus else (b_plus, a_plus)

    # vectorised branch selection -- selects a and b according to skew sign
    cond = a_plus > b_plus if want_negative_skew else a_plus < b_plus
    a = np.where(cond, a_plus, b_plus)
    b = np.where(cond, b_plus, a_plus)
    return a, b


def central_moments_beta(a, b):
    """
    Helper function: finds the central moments of the Beta distribution
    given shape parameters a (alpha) and b (beta).
    """
    mean_x = a / (a + b)
    var_x = a * b / ((a + b) ** 2 * (a + b + 1))
    return mean_x, var_x


# ====================
#
# ====================
def solve_beta_parameters_vec(tilnu_21, tilnu_31, tilnu_41, I_arr):
    """
    Vectorised: returns α,β,mean_x,c arrays with shape == I_arr.
    """
    I_arr = np.asarray(I_arr, dtype=float)
    tilnu_42 = 3.0 * tilnu_21**2
    beta1 = tilnu_31**2 / (tilnu_21**3 * I_arr)
    beta2 = (tilnu_42 * I_arr + tilnu_41) / (tilnu_21**2 * I_arr)

    u, v = analytic_u_v_from_betas(beta1, beta2)
    a, b = ab_from_u_v(u, v, want_negative_skew=(tilnu_31 < 0))

    mean_x, var_x = central_moments_beta(a, b)
    c_arr = np.sqrt((tilnu_21 * I_arr) / var_x)  # Scale (scales distribution)
    return a, b, mean_x, c_arr


def random_from_type1(tilnu_21, tilnu_31, tilnu_41, I_arr, rng=None):
    """
    Generates a draw from a unique shifted/scaled
    Beta distribution per element of I_arr in an array of shape I_arr.
    The probability distribution for every intensity should match the target moments.
    """
    if rng is None:
        rng = np.random.default_rng()

    a_arr, b_arr, mu_arr, c_arr = solve_beta_parameters_vec(tilnu_21, tilnu_31, tilnu_41, I_arr)

    y_raw = sp_beta.rvs(a_arr, b_arr, random_state=rng)
    return c_arr * (y_raw - mu_arr)


# ============================================================
# ============================================================


# Pearson Type 3 (aka Gamma) distribution


# ============================================================
# ============================================================


def random_from_type3(
    tilnu_21: float,
    tilnu_31: float,
    I_vals: np.ndarray,
    rng=None,
):
    """
    Generates a draw from a unique shifted/scaled Gamma distribution
    per element of I_arr in an array of shape I_arr.
    The probability distribution for every intensity should match the target moments.
    """

    if rng is None:
        rng = np.random.default_rng()

    I_vals = np.asarray(I_vals, dtype=float)

    scale = abs(tilnu_31) / (2.0 * tilnu_21)  # common scale
    shapes = 4.0 * tilnu_21**3 * I_vals / tilnu_31**2  # Γ shape per-pixel
    shifts = shapes * scale
    sign = 1 if tilnu_31 > 0 else -1

    y = gamma.rvs(a=shapes, scale=1.0, random_state=rng)  # raw Γ draws
    return _to_x(y, scale, shifts, sign)


# ============================================================
# ============================================================


# Pearson Type 4 distribution


# ============================================================
# ============================================================

# Used to avoid math overflow errors
_LOG2, _LOGPI = math.log(2.0), math.log(math.pi)


def _log_k(m: float, nu: float, a: float = 1.0):
    """
    Generates log(k).

    2^(2m-2) and Gamma terms blow up very fast.

    log(k) instead of "k" necessary to avoid math overflor errors.
    """
    return (
        (2 * m - 2) * _LOG2 + 2 * loggamma(m + 0.5j * nu).real - (_LOGPI + math.log(a) + loggamma(2 * m - 1))
    )


# ============================================================
# ============================================================

# Creating the comparison function for the PT4 acceptance-rejection sampler

# ============================================================
# ============================================================


def _log_reg_lower_gamma(a: float, x: float) -> float:
    """log P(a,x): regularized lower incomplete gamma, in a stable branch."""
    P = gammainc(a, x)
    if P == 0.0:
        return -np.inf
    if P < 0.5:
        return np.log(P)
    Q = gammaincc(a, x)  # = 1 - P
    if Q == 0.0:
        return 0.0
    return np.log1p(-Q)


def _branch_masses(m: float) -> Tuple[float, float, float]:
    """
    Return (g0, w_left, w_right) for normalized g(s), computed in log-space.
      g0 is the normalization constant
      P_left  = 0.5 * B(1/2, m-1/2)
      P_right = exp(2m) * (2m)^(-(2m-1)) * Γ(2m-1) * P(2m-1, 2m)
    """
    if not (m > 0.5):
        raise ValueError("Require m > 1/2 for normalizability.")
    a = 2.0 * m - 1.0
    log_P_left = np.log(0.5) + betaln(0.5, m - 0.5)
    log_P_right = (
        (2.0 * m) - (2.0 * m - 1.0) * np.log(2.0 * m) + loggamma(a) + _log_reg_lower_gamma(a, 2.0 * m)
    )
    logZ = logsumexp([log_P_left, log_P_right])
    g0 = np.exp(-logZ)
    w_left = np.exp(log_P_left - logZ)
    w_right = 1.0 - w_left
    return g0, w_left, w_right


# ========= proposal g(s) sampler =========


def _sample_left(n: int, m: float, rng: np.random.Generator) -> np.ndarray:
    """
    Left branch: s<0 with kernel (1+s^2)^(-m).
    Negative half of a scaled Student-t with ν=2m-1.
    """
    nu = 2 * m - 1
    Tdraw = t.rvs(df=nu, size=n, random_state=rng)
    S = -np.abs(Tdraw / np.sqrt(nu))  # reflect to s<0
    return S


def _sample_right(n: int, m: float, rng: np.random.Generator) -> np.ndarray:
    """
    Right branch: s>0 with kernel (1+s)^(-2m) * exp( 2m*s/(1+s) ).
    With y=1+s (>1) it's InvGamma(α=2m-1, β=2m), truncated to y>1.
    Exact draw via inverse survival function.
    """
    alpha = 2 * m - 1
    beta = 2 * m
    tiny = np.nextafter(0.0, 1.0)
    logS1 = invgamma.logsf(1.0, a=alpha, scale=beta)  # log P(Y>1)
    U = np.maximum(rng.random(n), tiny)
    Y = invgamma.isf(np.maximum(np.exp(logS1 + np.log(U)), tiny), a=alpha, scale=beta)
    return Y - 1.0  # s = y-1


def sample_g(
    m: float, size: ArrayLike = 1, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, float]:
    """
    g0 is normalization const
    Draw samples from normalized g(s). Returns (samples, g0).
    g(s) = g0*(1+s^2)^(-m) for s<0;  g0*(1+s)^(-2m) exp(2ms/(1+s)) for s>0; continuous at 0.
    """
    if rng is None:
        rng = default_rng()
    N = int(np.prod(size)) if hasattr(size, "__iter__") else int(size)
    g0, w_left, _ = _branch_masses(m)
    left_mask = rng.random(N) < w_left
    n_left = int(left_mask.sum())
    n_right = N - n_left
    out = np.empty(N, dtype=float)
    if n_left:
        out[left_mask] = _sample_left(n_left, m, rng)
    if n_right:
        out[~left_mask] = _sample_right(n_right, m, rng)
    return out.reshape(size if hasattr(size, "__iter__") else (N,)), g0


def _s_transform_constants(m: float, nu: float, a: float):
    """Constants for s↔x mapping and f_S."""
    theta = nu / (2.0 * m)
    root = math.sqrt(1.0 + theta * theta)
    logk = _log_k(m, nu, a)
    log_dxds = math.log(a) + 0.5 * math.log1p(theta * theta)  # log|dx/ds|
    return theta, root, logk, log_dxds


def _log_fS_from_s(
    s: float, m: float, nu: float, theta: float, root: float, logk: float, log_dxds: float
) -> float:
    """
    log f_S(s) = log f_X(x(s)) + log|dx/ds| under the −ν convention:
      f_X(x) ∝ (1+ξ^2)^(-m) * exp( -ν * atan ξ ),  ξ=(x-λ)/a
    Note: λ cancels in f_S(s); only m,ν,a matter in s-space.
    """
    xi = root * s - theta
    return logk + log_dxds - m * math.log1p(xi * xi) - nu * math.atan(xi)


def _log_g_pdf_scalar(s: float, m: float, g0: float) -> float:
    """log g_S(s) at scalar s (normalized g)."""
    if s < 0.0:
        return math.log(g0) - m * math.log1p(s * s)
    elif s > 0.0:
        return math.log(g0) - 2 * m * math.log1p(s) + (2 * m * s) / (1.0 + s)
    else:
        return math.log(g0)


def _peak_scale_logc(m: float, nu: float, a: float, g0: float) -> float:
    """
    Peak scaling: ensure envelope matches/exceeds PT4 at the mode s=0 (for −ν).
    Returns logc with c = exp(logc).
    """
    theta, root, logk, log_dxds = _s_transform_constants(m, nu, a)
    log_fs0 = _log_fS_from_s(0.0, m, nu, theta, root, logk, log_dxds)
    log_gs0 = math.log(g0)  # g_S(0) = g0
    return max(0.0, log_fs0 - log_gs0)


def pt4_rvs_devroye(m: float, nu: float, *, a: float = 1.0, lam: float = 0.0, size=None, rng=None):
    """
    Generates a random variate from specified Pearson Type 4 using a Devroye sampler.

    Sampler from
    Heinrich, *A Guide to the Pearson Type IV Distribution* (2004), Section 7
    """
    if rng is None:
        rng = np.random.default_rng()
    if m <= 1:
        raise ValueError("Pearson-IV generator requires m>1")

    b = 2 * m - 2
    M = math.atan2(-nu, b)
    cosM = b / math.hypot(b, nu)
    r_const = b * math.log(cosM) - nu * M
    rc = math.exp(-r_const - _log_k(m, nu, a))

    # Devroye sampler
    def _single():
        while True:
            z, x = 0.0, 4 * rng.random()
            s = 0
            if x > 2:
                x, s = x - 2, 1
            if x > 1:
                z, x = math.log(x - 1), 1 - math.log(x - 1)
            x = (M + rc * x) if s else (M - rc * x)
            if abs(x) >= math.pi / 2:
                continue
            if z + math.log(rng.random()) > b * math.log(math.cos(x)) - nu * x - r_const:
                continue
            return a * math.tan(x) + lam

    if size is None:
        return _single()
    out = np.empty(size, float)
    for i in range(out.size):
        out[i] = _single()
    return out.reshape(size)


def pt4_rvs_ar(m: float, nu: float, a: float, lam: float, rng: Generator | None = None) -> float:
    """
    ONE draw X ~ Pearson-IV (−ν convention) via AR with proposal g(s).
    Envelope is c*g(s) with c = exp(logc) chosen by peak-scaling at s=0.
    """
    if m <= 0.5:
        raise ValueError("Pearson-IV (and proposal) require m > 1/2.")
    if rng is None:
        rng = default_rng()

    theta, root, logk, log_dxds = _s_transform_constants(m, nu, a)
    g0, _, _ = _branch_masses(m)
    logc = _peak_scale_logc(m, nu, a, g0)

    # Required for comparison function > PT4
    flip = nu > 0.0

    while True:
        s0, _ = sample_g(m, size=1, rng=rng)
        s0 = float(s0[0])

        sg = s0

        if flip:
            s = -s0
        else:
            s = s0

        log_fS = _log_fS_from_s(s, m, nu, theta, root, logk, log_dxds)
        log_gS = _log_g_pdf_scalar(sg, m, g0)

        log_alpha = log_fS - log_gS - logc
        p = 1.0 if log_alpha >= 0.0 else math.exp(log_alpha)
        if rng.random() < p:
            xi = root * s - theta
            return a * xi + lam


def devroye_acc_rate(nu, a, m):
    """Analytical approximation of Devroye Pearson Type 4 sampler:"""

    b = 2 * m - 2
    M = math.atan2(-nu, b)
    cosM = b / math.hypot(b, nu)
    r_const = b * math.log(cosM) - nu * M
    rc = math.exp(-r_const - _log_k(m, nu, a))

    accept_1 = np.pi / (4 * rc)
    accept_2 = np.sqrt(2 / (np.pi * (2 * m + nu**2 / (2 * m))))
    return accept_1 * accept_2


def random_from_type4(
    tilnu21: float, tilnu31: float, tilnu41: float, I_array, *, devroye_threshold=0.005, rng=None
):
    """
    Generates a draw from a unique Pearson Type 4 per element of I_arr in an array of shape I_arr.
    The probability distribution for every intensity should match the target moments.

    If the acceptance rate of the devroye sampler > devroye_threshold, we use the devroye sampler.
    Otherwise, we use the AR sampler.
    """

    if rng is None:
        rng = np.random.default_rng()
    I_arr = np.asarray(I_array, dtype=float)
    flat_I = I_arr.ravel()

    # --- vectorised parameter computation ---
    tilnu42 = 3 * tilnu21**2
    mu2 = tilnu21 * flat_I
    beta1 = tilnu31**2 / (tilnu21**3 * flat_I)
    beta2 = (tilnu42 * flat_I + tilnu41) / (tilnu21**2 * flat_I)

    denom = 2 * beta2 - 3 * beta1 - 6
    r = 6 * (beta2 - beta1 - 1) / denom
    inner = 16 * (r - 1) - beta1 * (r - 2) ** 2  # Whatever's inside the sqrt

    if np.any(r <= 1) or np.any(inner <= 0):
        raise ValueError("Some intensities give invalid Pearson-IV parameters.")

    nu_mag = r * (r - 2) * np.sqrt(beta1) / np.sqrt(inner)
    sign = -1.0 if tilnu31 >= 0 else 1.0  # sign(µ3)=−sign(ν)
    nu = sign * nu_mag

    a = np.sqrt(mu2 * inner) / 4
    m = r / 2 + 1
    lam = a * nu / (2 * (m - 1))

    # Choose sampler based off of the Devroye acceptance rate
    acc_pred = np.array([devroye_acc_rate(nu_i, a_i, m_i) for nu_i, a_i, m_i in zip(nu, a, m, strict=False)])
    use_devroye = acc_pred > devroye_threshold

    draws = np.empty_like(flat_I)

    ## Draw from saplers using list comprehension
    # Devroye sampler
    for idx in np.where(use_devroye)[0]:
        draws[idx] = pt4_rvs_devroye(m[idx], nu[idx], a=a[idx], lam=lam[idx], rng=rng)
    # Acceptance-rejection sampler
    for idx in np.where(~use_devroye)[0]:
        draws[idx] = pt4_rvs_ar(m[idx], nu[idx], a=a[idx], lam=lam[idx], rng=rng)

    return draws.reshape(I_arr.shape)


# ============================================================
# ============================================================


# Pearson Type 5 (inverse-gamma) distribution


# ============================================================
# ============================================================


# DOCUMENT BETTER
def solve_pearson5_parameters_vec(tilnu_21, tilnu_31, I_arr):
    """
    Vectorised version of the analytic solver.

    Parameters
    ----------
    tilnu_21, tilnu_31 : floats
    I_arr              : ndarray of intensities

    Returns
    -------
    a_arr  : α array (shape parameters)
    b_arr  : β array (scale parameters)
    mu_arr : μ array (mean shifts)
    """
    I_arr = np.asarray(I_arr, dtype=float)

    beta1 = tilnu_31**2 / (tilnu_21**3 * I_arr)
    sqrt_t = np.sqrt(4.0 + beta1)

    p_plus = 4.0 * (1 + 2 / beta1 + sqrt_t / beta1)
    p_minus = 4.0 * (1 + 2 / beta1 - sqrt_t / beta1)
    p_arr = np.where(p_plus > 4.0, p_plus, p_minus)  # choose p > 4 element-wise

    sigma = np.sqrt(tilnu_21 * I_arr)
    gamma_p5 = sigma * (p_arr - 2) * np.sqrt(p_arr - 3)  # Pearson type 5 parameter

    a_arr = p_arr - 1.0  # α
    b_arr = gamma_p5  # β
    mu_arr = b_arr / (a_arr - 1.0)  # μ

    return a_arr, b_arr, mu_arr


def random_from_type5(tilnu_21, tilnu_31, I_arr, rng=None):
    """
    Generates a unique shifted inverse Gamma draw
    per element of I_arr in an array of shape I_arr.
    The probability distribution for every intensity
    should match the target moments.
    """
    if rng is None:
        rng = np.random.default_rng()

    a_arr, b_arr, mu_arr = solve_pearson5_parameters_vec(tilnu_21, tilnu_31, I_arr)

    y = invgamma.rvs(a=a_arr, scale=b_arr, random_state=rng)

    sign_scalar = 1.0 if tilnu_31 >= 0 else -1.0
    x = _to_x(y, scale=1.0, shift=mu_arr, sign=sign_scalar)
    return x


# ============================================================
# ============================================================


# Pearson Type 6 beta-prime distribution


# ============================================================
# ============================================================


def solve_pearson6_params(tilnu_21, tilnu_31, tilnu_41, I):
    """
    Parameters
    ----------
    tilnu_21, tilnu_31, tilnu_41 : floats
        Tilde-ν coefficients (shared across all intensities).
    I : array-like
        Noise-intensity values.

    Returns
    -------
    alpha, beta, scale, shift : ndarrays (shape = I.shape)
    sign : float  (+1 or −1) – skew sign (same for every element)
    """
    I = np.asarray(I, dtype=float)
    tilnu_42 = 3 * tilnu_21**2

    beta1 = tilnu_31**2 / (tilnu_21**3 * I)
    beta2 = (tilnu_41 + tilnu_42 * I) / (tilnu_21**2 * I)
    sign = 1.0 if tilnu_31 >= 0 else -1.0

    r = 6 * (beta2 - beta1 - 1) / (3 * beta1 - 2 * beta2 + 6)
    eps = r**2 / (4 + (beta1 / 4) * (r + 2) ** 2 / (r + 1))
    d = np.sqrt(r**2 - 4 * eps)

    q1 = (2 - r + d) / 2
    q2 = (r - 2 + d) / 2
    alpha = q2 + 1
    beta = q1 - q2 - 1

    var1 = alpha * (alpha + beta - 1) / ((beta - 2) * (beta - 1) ** 2)
    scale = np.sqrt(tilnu_21 * I / var1)
    mean1 = alpha / (beta - 1)
    shift = scale * mean1

    return alpha, beta, scale, shift, sign


def random_from_type6(tilnu_21, tilnu_31, tilnu_41, I, rng=None):
    """
    Generates a draw from a unique shifted/scaled
    Beta Prime  per element of I_arr in an array of shape I_arr.
    The probability distribution for every intensity should match the target moments.
    """

    if rng is None:
        rng = np.random.default_rng()

    a, b, s, shift, sgn = solve_pearson6_params(tilnu_21, tilnu_31, tilnu_41, I)

    y = betaprime.rvs(a, b, random_state=rng)  # one deviate per element (SciPy 1.11+)
    return _to_x(y, s, shift, sgn)
