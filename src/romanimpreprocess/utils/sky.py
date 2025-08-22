"""
Utilities for simple sky estimation.

Functions
---------
binkxk
    Bin-averaging utility for 2D array, kxk.
smooth_mode
    Find the mode of the smoothed histogram.
medfit
    Fits a low-order polynomial to a 2D array.

"""

import numpy as np
import scipy.stats
from scipy.special import legendre_p


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


def medfit(arr, N=8, order=2):
    """
    Fits a low-order polynomial to a 2D array.

    Inputs
    -------
    arr : np.array
        The 2D image.
    N : int, optional
        Number of regions to break into on each dimension
        (so total is N^2).
    order : int, optional
        Order of polynomial to fit.

    Returns
    -------
    coef : np.array
        The flattened array of polynomial coefficients (see Notes for ordering).
    arrmed : np.array
        The fit to the median (same shape as `arr`).

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

    # now get the medians
    meds = np.nanmedian(arr[py : py + N * ky, px : px + N * kx].reshape((N, ky, N, kx)), axis=(1, 3))

    # and now get the coefficients
    # this is not efficient for large N, but we don't anticipate such a use case
    nc = (order + 1) * (order + 2) // 2
    basis = np.zeros((nc, N, N))
    k = 0
    for i in range(order + 1):
        temp = legendre_p(i, u)
        for j in range(order + 1 - i):
            basis[k, :, :] = temp * legendre_p(j, v)
            k += 1

    # now build the linear system solution, Ax=b
    A = np.zeros((nc, nc))
    b = np.zeros(nc)
    for ipix in range(N):
        for jpix in range(N):
            if not np.isnan(meds[jpix, ipix]):
                A += np.outer(basis[:, jpix, ipix], basis[:, jpix, ipix])
                b += meds[jpix, ipix] * basis[:, jpix, ipix]
    x = np.linalg.solve(A, b)

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
    return coef, arrmed.astype(arr.dtype)
