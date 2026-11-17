"""
Utilities for simple sky estimation.

Functions
---------
binkxk
    Bin-averaging utility for 2D array, kxk.
smooth_mode
    Find the mode of the smoothed histogram.
medfit
    Fits a low-order polynomial to a 2D array (with optional masking and outlier-box rejection).
moransi_sky_mask
    Source mask based on the Moran's I statistic (True = sky); wraps ``moransi_sourcemasking``.

"""

import warnings

import numpy as np
import scipy.stats
from scipy.ndimage import binary_dilation
from scipy.special import legendre_p
try:
    from moransi_sourcemasking import make_sourcemask
except ImportError as err:
    raise ImportError(
        f"Moran's I sky masking requires the moransi_sourcemasking package"
    ) from err


def binkxk(arr, k):
    """
    Bin-averaging utility for 2D array, kxk.

    Parameters
    ----------
    arr : np.array
        2D array
    k : int
        Bin every `k` pixels on both axes.

    Returns
    -------
    np.array of float
        2D array, reduced by a factor of `k` on each axis.
        The "remainder" pixels (if any) are ignored.

    """

    (ny, nx) = np.shape(arr)
    nyo = ny // k
    nxo = nx // k

    return np.mean(arr[: k * nyo, : k * nxo].reshape((nyo, k, nxo, k)), axis=(1, 3))


def smooth_mode(arr, pc=25.0, pksmooth=0.5, niter=3):
    """
    Find the mode of the smoothed histogram.

    Ignores nans.

    Parameters
    ----------
    arr : np.array
        The image from which we want to take the mode.
    pc : float, optional
        Percentile cut for the histogram.
    pksmooth : float, optional
        Number of sigmas to smooth.
    niter : int, optional
        Number of peak-finding iterations.

    Returns
    --------
    (float, float)
        Best fit mode and width of weighting function.

    """

    # initial setup of the center and sigma of the distribution
    c1 = np.nanpercentile(arr, pc)
    c2 = np.nanpercentile(arr, 50.0)
    c3 = np.nanpercentile(arr, 100.0 - pc)
    gauss_iqr_in_sigmas = scipy.stats.norm.ppf((100.0 - pc) / 100.0) * 2
    ctr = c2
    sigma = (c3 - c1) / gauss_iqr_in_sigmas

    N = 21
    for _ in range(niter):
        histsmooth = np.zeros(N)
        z = ctr + np.linspace(-1, 1, N) * sigma
        for i in range(1, N - 1):
            w = np.exp(-0.5 * ((z[i] - arr) / (pksmooth * sigma)) ** 2)
            histsmooth[i] = np.sum(np.where(np.isnan(w), 0.0, w))

        # now find the peak of the histogram
        i_peak = np.argmax(histsmooth)
        # now we want to fit a quadratic to these 3 points
        b = (histsmooth[i_peak + 1] - histsmooth[i_peak - 1]) / 2.0
        a = (histsmooth[i_peak + 1] + histsmooth[i_peak - 1]) / 2.0 - histsmooth[i_peak]
        ctr = z[i_peak] + (z[1] - z[0]) * (-b / 2.0 / a)

    return (ctr, sigma * pksmooth)


def _legendre_basis(u, v, order):
    """2D Legendre basis P_i(u) P_j(v), i+j <= order, ordered as in `medfit`."""

    nc = (order + 1) * (order + 2) // 2
    basis = np.zeros((nc,) + np.shape(u))
    k = 0
    for i in range(order + 1):
        temp = legendre_p(i, u)
        for j in range(order + 1 - i):
            basis[k] = temp * legendre_p(j, v)
            k += 1
    return basis


def medfit(
    arr,
    N=8,
    order=2,
    mask=None,
    fit_clip=None,
    fit_maxiter=10,
    min_good_frac=0.1,
    return_info=False,
):
    """
    Fits a low-order polynomial to a 2D array.

    The image is broken into N x N boxes; a robust level is computed in each box,
    and a 2D Legendre polynomial is fit to the box levels.

    Inputs
    -------
    arr : np.array
        The 2D image.
    N : int, optional
        Number of regions to break into on each dimension
        (so total is N^2).
    order : int, optional
        Order of polynomial to fit.
    mask : np.array of bool, optional
        Pixels to exclude (True = excluded), e.g., a source mask.
        Must be the same shape as `arr`. NaNs in `arr` are always excluded.
    fit_clip : float, optional
        If given, the polynomial fit is iterated, rejecting boxes whose residual
        from the fit exceeds `fit_clip` robust standard deviations (of the box
        residuals). Rejected boxes are not used in the final fit.
    fit_maxiter : int, optional
        Maximum number of rejection iterations in the polynomial fit.
    min_good_frac : float, optional
        Boxes with fewer than this fraction of usable pixels are dropped.
        Only applied if `mask` is given (otherwise the original behavior,
        using all non-NaN pixels, is preserved).
    return_info : bool, optional
        If True, also return a dictionary of diagnostics.

    Returns
    -------
    coef : np.array
        The flattened array of polynomial coefficients (see Notes for ordering).
    arrmed : np.array
        The fit to the median (same shape as `arr`).
    info : dict
        Only if `return_info` is True. Contains ``meds`` (N x N box levels),
        ``used`` (N x N bool, boxes used in the final fit), ``ngood`` (usable
        pixels per box), and ``niter`` (number of fit iterations).

    Notes
    -----
    The polynomial fit is of the form (in pseudocode)::

      u = 2*x/nx-1
      v = 2*y/ny-1
      arrmed[y,x] = sum coef_ij P_i(u) P_j(v)

    (so u and v are scaled to the range -1 to +1).

    The ordering in `coef` is::

      0,0   0,1   0,2   ...   0,order-1   0,order
      1,0   1,1   1,2   ...   1,order-1
      ...
      order,0

    The total number of coefficients is ``(order+1)*(order+2)//2``.

    With the default arguments (no mask, no clipping), this is a plain median
    in each box followed by an unweighted least-squares fit.

    """

    # get indices for the center, as close as we can get
    (ny, nx) = np.shape(arr)
    kx = nx // N
    ky = ny // N
    px = (nx % N) // 2
    py = (ny % N) // 2
    u_ = 2 * (px - 0.5 + kx * np.linspace(0.5, N - 0.5, N)) / nx - 1
    v_ = 2 * (py - 0.5 + ky * np.linspace(0.5, N - 0.5, N)) / ny - 1
    u, v = np.meshgrid(u_, v_)

    # now get the box levels
    sub = np.asarray(arr[py : py + N * ky, px : px + N * kx], dtype=np.float64)
    if mask is not None:
        if np.shape(mask) != np.shape(arr):
            raise ValueError(f"mask shape {np.shape(mask)} does not match image shape {np.shape(arr)}")
        sub = np.where(mask[py : py + N * ky, px : px + N * kx], np.nan, sub)
    boxes = sub.reshape((N, ky, N, kx)).transpose((0, 2, 1, 3)).reshape((N, N, ky * kx))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fully masked boxes give NaN
        meds = np.nanmedian(boxes, axis=-1)
    ngood = np.sum(np.isfinite(boxes), axis=-1)
    if mask is not None:
        meds = np.where(ngood >= min_good_frac * kx * ky, meds, np.nan)

    # and now get the coefficients
    nc = (order + 1) * (order + 2) // 2
    basis = _legendre_basis(u, v, order)
    used = np.isfinite(meds)
    if np.count_nonzero(used) < nc:
        raise ValueError(f"medfit: only {np.count_nonzero(used)} usable boxes for {nc} coefficients")

    # now build the linear system solution, Ax=b (over the boxes in use);
    # with fit_clip, iteratively reject outlier boxes and re-solve
    niter = 0
    while True:
        A = np.zeros((nc, nc))
        b = np.zeros(nc)
        for ipix in range(N):
            for jpix in range(N):
                if used[jpix, ipix]:
                    A += np.outer(basis[:, jpix, ipix], basis[:, jpix, ipix])
                    b += meds[jpix, ipix] * basis[:, jpix, ipix]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as err:
            jj, ii = np.nonzero(used)
            raise ValueError(
                f"medfit: singular fit (order={order}) with {len(ii)} boxes in use at "
                f"(row, col) = {list(zip(jj.tolist(), ii.tolist()))}"
            ) from err
        if fit_clip is None or niter >= fit_maxiter:
            break
        niter += 1
        resid = meds - np.tensordot(x, basis, axes=1)
        r_used = resid[used]
        sig = 1.482602218505602 * np.median(np.abs(r_used - np.median(r_used)))
        newused = np.isfinite(meds) & (np.abs(resid) <= fit_clip * sig) if sig > 0 else used
        if np.count_nonzero(newused) < nc or np.array_equal(newused, used):
            break
        used = newused

    # now get the Legendre polynomials on a pixel grid
    LPX = np.zeros((order + 1, nx))
    LPY = np.zeros((order + 1, ny))
    u_ = np.linspace(-1, 1 - 2 / nx, nx)
    v_ = np.linspace(-1, 1 - 2 / ny, ny)
    for i in range(order + 1):
        LPX[i, :] = legendre_p(i, u_)
    for j in range(order + 1):
        LPY[j, :] = legendre_p(j, v_)

    # and this is where the components are summed up
    arrmed = np.zeros((ny, nx))
    k = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            arrmed += x[k] * np.outer(LPY[j, :], LPX[i, :])
            k += 1

    coef = x
    if return_info:
        return coef, arrmed.astype(arr.dtype), {"meds": meds, "used": used, "ngood": ngood, "niter": niter}
    return coef, arrmed.astype(arr.dtype)




def moransi_sky_mask(arr, bad=None, weight=None, dilate=0, **pars):
    """
    Source mask based on the sliding-patch Moran's I statistic.

    This is a thin wrapper around ``make_sourcemask`` from H. Ferguson's
    ``moransi_sourcemasking`` package (https://github.com/hcferguson/moransi_sourcemasking),
    which is an optional dependency: it is only imported when this function is called.

    Parameters
    ----------
    arr : np.array
        The 2D image.
    bad : np.array of bool, optional
        Pixels to ignore (True = bad), e.g., from the DQ array.
    weight : np.array, optional
        Inverse-variance weights.
    dilate : int, optional
        Number of pixels by which to grow the source mask (applied after ``make_sourcemask``).
    **pars
        Control parameters passed to ``make_sourcemask`` (e.g., ``patch_size``, ``kernel_width``,
        ``n_box_passes``, ``opening_iterations``, ``threshold_type``, ``threshold_value``).
        Unspecified parameters take that package's defaults. Note that with
        ``threshold_type="percentage"``, ``threshold_value`` is the percentage of pixels
        with the highest I that are flagged as *sources*.

    Returns
    -------
    np.array of bool
        True for sky pixels, False for sources and for bad or non-finite pixels.

    """


    valid = np.isfinite(arr)
    if bad is not None:
        valid &= ~np.asarray(bad, dtype=bool)
    is_sky = make_sourcemask(arr, pars if pars else None, bad_mask=~valid, weight=weight)
    if dilate > 0:
        is_sky = ~binary_dilation(~is_sky, iterations=dilate)
    return is_sky & valid
