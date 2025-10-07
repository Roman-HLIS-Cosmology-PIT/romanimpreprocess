"""
Utility functions for ramp fitting.

Functions
---------
construct_weights
    Makes a numpy array of weights for slope fitting.
jump_detect
    Searches for a jump.
ramp_fit
    Ramp fitting.

"""

import asdf
import numpy as np
from roman_datamodels.dqflags import pixel


def construct_weights(u, meta, exclude_first=True):
    """
    Makes a numpy array of weights for slope fitting.

    Parameters
    ----------
    u : float
        Poisson to read noise ratio, unit: 1/(e*s).
    meta : dict
        Metadata for setting the weights.
    exclude_first : bool, optional
        Whether to exclude the first read (this is because the reset-read frame
        is sometimes hard to calibrate).

    Returns
    -------
    K : np.array of float
        The weight vector; length is the number of groups (``meta['ngrp']``).

    Notes
    -----
    These are optimized for a ratio of u = flux / (gain * read_noise_variance).
    The units are: [DN/s] / ([e/DN] * [DN^2]) = 1/(e*s).

    The `meta` dictionary should contain:
    * ``'ngrp'`` : int
      Number of groups
    * ``'N'`` : np.array of int
      Number of frames in each group
    * ``'tbar'`` : np.array of float
      Mean time since reset of each group
    * ``'tau'`` : np.array of float
      Variance-weighted time since reset of each group

    This follows the notation and optimization of Casertano et al. (2022),
    except that we don't use the adaptive aspect of the algorithm.

    The returned array `K` is intended to be used in computing a slope,
    sum_i K_i R_i (in DN/s).
    The weights should sum to 0 so that we aren't sensitive to the reset level.

    """

    K = np.zeros(meta["ngrp"])
    start = 0
    if exclude_first:
        start = 1
    ngrp = meta["ngrp"] - start

    # for the matrix algebra, we go up to 64 bits
    tbar = meta["tbar"][start:].astype(np.float64)
    tau = meta["tau"][start:].astype(np.float64)
    C = np.zeros((ngrp, ngrp))
    for i in range(ngrp):
        C[i, i] = 1.0 / meta["N"][start + i] + u * tau[i]
        for j in range(i):
            C[i, j] = C[j, i] = u * tbar[j]
    W = np.linalg.inv(C)
    Ws = np.sum(W, axis=0)
    Wt = W @ tbar
    F0 = np.sum(W)
    F1 = np.sum(Wt)
    F2 = np.dot(tbar, Wt)
    D = F0 * F2 - F1**2
    K[start:] = (F0 * Wt - F1 * Ws) / D

    return K.astype(np.float32)


def jump_detect(data, rdq, pdq, meta, caldir, mylog, exclude_first=True, truncate_ramp=None):
    """
    Searches for a jump.

    Note that affected pixels are only flagged by this function: they are not corrected!

    Parameters
    ----------
    data : np.array
        The input data in DN, shape = (ngrp,ny,nx).
    rdq : np.array
        3D array, flags (ramp data quality)
    pdq : np.array
        2D array, flags (pixel data quality)
    meta : dict
        Other metadata (right now: frame_time and read_pattern)
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    exclude_first : bool, optional
        Exclude the first sample?
    truncate_ramp : int or None, optional
        If given, truncates the ramp at this sample (useful for saturated pixels).

    Returns
    -------
    slope : np.array
        Slope image (2D, DN/s).
    slope_err_read : np.array
        Slope error image from read noise (2D, DN/s).
    slope_err_poisson : np.array
        Slope error image from Poisson noise (2D, DN/s).
    smap : np.array
        3D jump significance cube, dimensionless (axes: difference, y, x).

    Notes
    -----
    Data operated on is in the cube data (ngrp,ny,nx); and the uint32 flags are in
    rdq (ngrp,ny,nx) and pdq (ny,nx).

    The `meta` dictionary contains at least:

    * ``'ngrp'`` : int
      Number of groups
    * ``'N'`` : np.array of int
      Number of frames in each group
    * ``'tbar'`` : np.array of float
      Mean time since reset of each group
    * ``'tau'`` : np.array of float
      Variance-weighted time since reset of each group
    * ``'K'`` : np.array
      1D weight vector, ususally comes from construct_weights.

    If `meta` contains "jump_detect_pars", then this can override the default
    settings in the algorithm.

    The gain is read from `caldir`, and is in e/DN.

    The current version implements the Sharma & Casertano (2024) flagging algorithm
    except for fitting the weights.

    See: Sanjib Sharma and Stefano Casertano 2024 PASP 136 054504
    DOI 10.1088/1538-3873/ad4b9e

    """

    # get basic information
    ngrp = meta["ngrp"]
    (ny, nx) = np.shape(pdq)
    start = 0
    if exclude_first:
        start = 1
    K = meta["K"]

    # truncation -- this is for bright sources so will use the first and last points
    if truncate_ramp is not None:
        ngrp = truncate_ramp
        K = np.zeros(ngrp, dtype=np.float32)
        K[-1] = 1.0 / (meta["tbar"][ngrp - 1] - meta["tbar"][start])
        K[start] = -K[-1]

    # get detection parameters
    SthreshA = 5.5
    SthreshB = 4.5
    IthreshA = 1.0
    IthreshB = 1000.0
    if "jump_detect_pars" in meta:
        if "SthreshA" in meta["jump_detect_pars"]:
            SthreshA = float(meta["jump_detect_pars"]["SthreshA"])
        if "SthreshB" in meta["jump_detect_pars"]:
            SthreshB = float(meta["jump_detect_pars"]["SthreshB"])
        if "IthreshA" in meta["jump_detect_pars"]:
            IthreshA = float(meta["jump_detect_pars"]["IthreshA"])
        if "IthreshB" in meta["jump_detect_pars"]:
            IthreshB = float(meta["jump_detect_pars"]["IthreshB"])

    # get slope
    slope = np.einsum("t,tij->ij", K, data[:ngrp, :, :] - data[1, :, :][None, :, :]).astype(np.float32)

    # jump detection map
    smap = np.zeros((2 * (ngrp - start) - 3, ny, nx), dtype=np.float32)

    # get Poisson variance information
    # first coef (units: 1/time, since K has units of 1/time)
    # this would be 1/exposure time for simple CDS
    coef = 0.0
    for i in range(ngrp - start):
        coef += K[i] ** 2 * meta["tau"][i + start]
        for j in range(i):
            coef += 2.0 * K[i] * K[j] * meta["tbar"][j + start]
    with asdf.open(caldir["gain"]) as f:
        dvardt = np.clip(slope / np.clip(f["roman"]["data"], 1e-4, 1e4), 0.0, None)
        # Poisson variance [DN^2] per second, clipped to be positive and avoid divbyzero
        slope_err_poisson = np.sqrt(np.clip(coef * dvardt, 0, None)).astype(np.float32)

    # get read noise variance information
    with asdf.open(caldir["read"]) as f:
        sig2read = f["roman"]["data"] ** 2  # single read noise [DN^2]
        slope_err_read = (f["roman"]["data"] * np.sqrt(np.sum(K**2 / np.array(meta["N"][:ngrp])))).astype(
            np.float32
        )
        # slope_err_read is a standard deviation [DN]

    # threshold map
    x = np.clip(slope, IthreshA, IthreshB)
    x = np.log(x / IthreshA) / np.log(IthreshB / IthreshA)
    sthresh = SthreshA + (SthreshB - SthreshA) * x
    nb = meta["nborder"]
    (ny, nx) = np.shape(x)
    del x
    mylog.append(f"median threshold for CR jump is {np.median(sthresh):f} sigma\n")
    mylog.append(f"truncate at {truncate_ramp}, K = {K}\n")

    sl = 0
    for i in range(start, ngrp - 1):
        dimax = 2
        if i == ngrp - 2 or ngrp - 1 - start == 2:
            dimax = 1
        for di in range(1, 1 + dimax):
            delta_slope = (data[i + di, :, :] - data[i, :, :]) / (
                meta["tbar"][i + di] - meta["tbar"][i]
            ) - slope
            w = np.zeros(ngrp)
            w[i + di] = 1.0 / (meta["tbar"][i + di] - meta["tbar"][i])
            w[i] = -1.0 / (meta["tbar"][i + di] - meta["tbar"][i])
            w -= K
            var_delta_slope = np.zeros((ny, nx))
            for a in range(ngrp):
                var_delta_slope += w[a] ** 2 * (dvardt * meta["tau"][a] + sig2read / np.array(meta["N"][a]))
                for b in range(a):
                    var_delta_slope += 2 * w[a] * w[b] * dvardt * meta["tbar"][b]
            smap[sl, :, :] = delta_slope / np.sqrt(var_delta_slope).astype(np.float32)
            mylog.append(
                f"cr_bkgnd: grp={i:d},{i+di:d}, medvar={np.median(var_delta_slope):f},"
                + f" 99%ile={np.percentile(smap[sl,:,:],99):f}\n"
            )

            # now the masking
            rdq[i, nb : ny - nb, nb : nx - nb] |= np.where(
                smap[sl, nb : ny - nb, nb : nx - nb] > sthresh[nb : ny - nb, nb : nx - nb], pixel.JUMP_DET, 0
            ).astype(np.uint32)

            sl += 1  # move to next slice

    return slope, slope_err_read, slope_err_poisson, smap


def ramp_fit(data, rdq, pdq, meta, caldir, mylog, exclude_first=True):
    """
    Ramp fitting.

    Note that the slope image still fits objects that saturate during the exposure, since
    the same bright stars will keep saturating and we don't want to keep masking them. But
    for CR hits on unsaturated objects we want to reject that exposure, since the ePSF may
    be different due to jitter.

    Parameters
    ----------
    data : np.array
        The input data in DN, shape = (ngrp,ny,nx).
    rdq : np.array
        3D array, flags (ramp data quality)
    pdq : np.array
        2D array, flags (pixel data quality)
    meta : dict
        Other metadata (right now: frame_time and read_pattern)
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    exclude_first : bool, optional
        Exclude the first sample?

    Returns
    -------
    slope : np.array
        Slope image (2D, DN/s).
    slope_err_read : np.array
        Slope error image from read noise (2D, DN/s).
    slope_err_poisson : np.array
        Slope error image from Poisson noise (2D, DN/s).

    Notes
    -----
    The `meta` dictionary contains at least:

    * ``'ngrp'`` : int
      Number of groups
    * ``'N'`` : np.array of int
      Number of frames in each group
    * ``'tbar'`` : np.array of float
      Mean time since reset of each group
    * ``'tau'`` : np.array of float
      Variance-weighted time since reset of each group
    * ``'K'`` : np.array
      1D weight vector, ususally comes from construct_weights.

    """

    loc_rdq = np.zeros_like(rdq)  # local copy since which pixels we want to flag will depend on saturation

    # basic ramp fit
    slope, slope_err_read, slope_err_poisson, smap = jump_detect(
        data, loc_rdq, pdq, meta, caldir, mylog, exclude_first, truncate_ramp=None
    )
    # fits.PrimaryHDU(smap).writeto('crdet.fits',overwrite=True) <-- for testing
    # update quality flags if not saturated
    unsat = ~rdq[-1, :, :] & pixel.SATURATED != 0
    rdq |= np.where(unsat[None, :, :], loc_rdq, 0)
    mylog.append(f"testing {np.count_nonzero(unsat):d} unsaturated pixels\n")

    # saturated pixels
    start = 0
    if exclude_first:
        start = 1
    for iend in range(meta["ngrp"] - 1, 2 + start, -1):
        # get pixels that saturate in this step
        thislayer = rdq[iend, :, :] & ~rdq[iend - 1, :, :] & pixel.SATURATED != 0
        loc_rdq[:, :, :] = 0  # reset the local data quality so we don't propagate flags from bad fits
        slope_, slope_err_read_, slope_err_poisson_, smap = jump_detect(
            data, loc_rdq, pdq, meta, caldir, mylog, exclude_first, truncate_ramp=iend
        )
        slope = np.where(thislayer, slope_, slope)
        slope_err_read = np.where(thislayer[:, :], slope_err_read_, slope_err_read)
        slope_err_poisson = np.where(thislayer[:, :], slope_err_poisson_, slope_err_poisson)
        rdq |= np.where(thislayer[None, :, :], loc_rdq, 0)
        mylog.append(f"testing {np.count_nonzero(thislayer):d} pixels saturated at group {iend:d}\n")

    # propagate flags
    pdq2 = np.zeros_like(pdq)
    pdq2 |= np.bitwise_or.reduce(np.where(~rdq & pixel.SATURATED, rdq, 0), axis=0)
    # do not use pixels that saturated too fast
    pdq2 |= np.where(rdq[1 + start, :, :] & pixel.SATURATED != 0, pixel.DO_NOT_USE, 0).astype(np.uint32)
    # now fully flag the pixels that saturated
    pdq2 |= np.bitwise_or.reduce(rdq & pixel.SATURATED, axis=0)
    # and write these to the original
    pdq |= np.where(~pdq & pixel.REFERENCE_PIXEL, pdq2, 0)

    return slope, slope_err_read, slope_err_poisson
