"""
Main tools to drive exposure level processing.

Functions
---------
wcs_from_config
    Extracts a WCS from the configuration file.
initializationstep
    Creates and initializes L2 data.
saturation_check
    Flags saturated pixels in a 3D cube.
subtract_dark_current
    Subtracts dark current in a 3D cube.
repackage_wcs
    Packages a WCS so that it can be handed to romanisim.
calibrateimage
    L1->L2 driver.

"""

import sys
import warnings  # noqa: F401

import asdf

# not actually doing a simulation but needed to pass around the WCS types
import galsim  # noqa: F401
import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits
from roman_datamodels import datamodels
from roman_datamodels.dqflags import group, pixel
from romancal.dark_current import dark_current_step
from romancal.dark_decay.dark_decay import subtract_dark_decay
from romancal.datamodels.fileio import open_dataset
from romancal.dq_init import dq_initialization
from romancal.ramp_fitting import ramp_fit_step
from romancal.saturation import saturation
from romancal.wfi18_transient.wfi18_transient import correct_anomaly
from romanisim import image as rimage
from romanisim import persistence as rip
from romanisim import wcs as riwcs

from .. import pars
from ..utils import (
    coordutils,
    fitting,
    flatutils,
    ipc_linearity,
    maskhandling,
    processlog,
    reference_subtraction,
    sky,
    typefix,
)

# local imports
from . import oututils

### function definitions below here


def wcs_from_config(config):
    """
    Gets a WCS object from the configuration.

    Currently supports FITS headers imported from a simulation.

    Parameters
    ----------
    config : dict
        Configuration dictionary (usually imported from YAML).

    Returns
    -------
    astropy.io.fits.header.Header
        The WCS as a FITS header.

    """

    if "FITSWCS" in config:
        with open(config["FITSWCS"]) as f:
            return fits.Header.fromstring(f.read())

    # if no WCS was found, just return None (we'll deal with this later)
    return None


def initializationstep(config, caldir, mylog):
    """
    Initialization step.

    Parameters
    ----------
    config : dict
        Configuration dictionary (usually imported from YAML).
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.

    Returns
    -------
    ramp_model : RampModel
        ramp data model including data, groupdq, pixeldq, metadata
    meta: dict
        Other metadata (right now: frame_time and read_pattern)
    """

    if "mask" in caldir:
        maskfile = asdf.open(caldir["mask"])
        mask = datamodels.MaskRefModel.create_from_model(maskfile["roman"])
    else:
        mask = None

    with open_dataset(config["IN"], update_version=True) as l1model:
        ramp_model = dq_initialization.do_dqinit(l1model, mask, expand_gw_flagging=1)

    if "mask" in caldir:
        maskfile.close()

    meta = {
        "frame_time": ramp_model.meta.exposure.frame_time,
        "read_pattern": ramp_model.meta.exposure["read_pattern"],
    }

    # more information
    meta["ngrp"] = len(meta["read_pattern"])
    meta["tbar"] = np.zeros(meta["ngrp"], dtype=np.float32)
    meta["tau"] = np.zeros(meta["ngrp"], dtype=np.float32)
    meta["N"] = np.zeros(meta["ngrp"], dtype=np.int16)
    for i in range(meta["ngrp"]):
        # N_i, tbar_i, and tau_i as defined in Casertano et al. 2022
        meta["N"][i] = len(meta["read_pattern"][i])
        t0 = meta["read_pattern"][i][0]
        meta["tbar"][i] = (t0 + (meta["N"][i] - 1) / 2.0) * meta["frame_time"]
        meta["tau"][i] = (t0 + (meta["N"][i] - 1) * (2 * meta["N"][i] - 1) / (6.0 * meta["N"][i])) * meta[
            "frame_time"
        ]

    if config.get("EXCLUDE_FIRST", True):
        ramp_model["groupdq"][0, ...] |= group.DO_NOT_USE

    return ramp_model, meta


def saturation_check(ramp_model, caldir, mylog, backup=1, skip_firstn=1):
    """
    Flags saturated pixels (in both 3D and 2D arrays).

    Performs a saturation check on the data cube (`data`) using the calibration files in `caldir`.
    Information is appended to `mylog`. The flags `rdq` and `pdq` are updated in place.

    This function serves as a wrapper for ``flag_saturation`` (imported from ``romancal``).

    Parameters
    ----------
    ramp_model : roman_datamodels.datamodels.RampModel
        data model including resultant cube
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    backup : int
        Number of resultants to "back up" when flagging saturation.
    skip_firstn : int
        Do not check the first n resultants in ramp_model.data for saturation.

    """

    with asdf.open(caldir["saturation"]) as satreffile:
        satref = datamodels.SaturationRefModel.create_from_model(satreffile["roman"])
        if skip_firstn != 0:
            old_data = ramp_model.data
            old_dq = ramp_model.groupdq
            old_read_pattern = ramp_model.meta.exposure.read_pattern
            ramp_model.data = old_data[skip_firstn:, ...]
            ramp_model.groupdq = old_dq[skip_firstn:, ...]
            ramp_model.meta.exposure.read_pattern = ramp_model.meta.exposure.read_pattern[skip_firstn:]
        saturation.flag_saturation(ramp_model, satref, n_pix_grow_sat=1, backup=backup)
        if skip_firstn != 0:
            ramp_model.data = old_data
            ramp_model.groupdq = old_dq
            ramp_model.meta.exposure.read_pattern = old_read_pattern


def subtract_dark_current(image_model, caldir, mylog):
    """
    Subtracts dark current from a rate image.

    image_model is updated in place.

    romancal expects to subtract from an active-region image only
    (4088x4088) rather than a full-frame 4096x4096 image, so there
    are some gymnastics to handle that difference.

    dark subtraction occurs after IPC deconvolution, but the dark
    reference file is IPC-convolved, so this routine also corrects
    the dark reference file for IPC.

    Parameters
    ----------
    image_model : roman_datamodels.datamodels.ImageModel
        2D Roman image model (DN / s), full 4096x4096 frame.
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    """

    nb = pars.nborder
    with asdf.open(caldir["dark"]) as f:
        darkref = datamodels.DarkRefModel.create_from_model(f["roman"])

        # correct IPC in dark reference file
        if "ipc4d" in caldir:
            dslope = np.array(darkref.dark_slope, dtype=np.float32)[None, :, :]
            ipc_linearity.correct_cube(dslope, caldir["ipc4d"], None, gain_file=caldir["gain"])
            darkref.dark_slope = dslope[0]
            mylog.append("IPC-corrected the dark slope\n")

        full_data = image_model.data
        full_dq = image_model.dq
        image_model.data = full_data[nb:-nb, nb:-nb]
        image_model.dq = full_dq[nb:-nb, nb:-nb]
        dark_current_step.subtract_dark_current(image_model, darkref)
        image_model.data = full_data
        image_model.dq = full_dq
    # Note: we deliberately do not mark image_model.meta.cal_step.dark here.
    # make_asdf rebuilds the output metadata and resets cal_step, so the marking
    # would not reach the output; cal_step handling is deferred until that is
    # reworked.


def repackage_wcs(thewcs):
    """
    Packages a WCS to feed to romanisim.

    Right now supports FITS-standard headers from a simulation.
    Since this for compatibility in ramp-fitting routines, can use this
    and overwrite the WCS in the L2 ASDF tree with a full-accuracy gwcs
    at a later stage.

    Parameters
    ----------
    thewcs : astropy.io.fits.Header or galsim.CelestialWCS
        Input WCS.

    Returns
    -------
    class
        Packaged WCS, 2 layers deep for compatibility with romanisim.

    """

    # make WCS --- a few ways of doing this
    while True:
        wcsobj = None

        class Blank:
            pass

        # first try a FITS header
        if isinstance(thewcs, fits.Header):
            wcsobj = Blank()
            wcsobj.header = Blank()
            wcsobj.header.header = thewcs
            break

        # should work if this is a GalSim WCS
        # I commented this option out since I think it has a bug related to the 0 vs 1 offset,
        # but we're currently not using it.
        # Make sure to test it if you un-comment this.
        # -C.H. 02/12/26
        #
        # try:
        #     header = fits.Header()
        #     thewcs.writeToFitsHeader(header, galsim.BoundsI(0, pars.nside_active, 0, pars.nside_active))
        #     # offset to FITS convention -- this is undone later
        #     header["CRPIX1"] += 1
        #     header["CRPIX2"] += 1
        #     wcsobj = Blank()
        #     wcsobj.header = Blank()
        #     wcsobj.header.header = header
        #     warnings.warn("Use of GalSim WCS in calibrate is not fully working yet!")
        #     break
        # except Exception as e:
        #     wcsobj = None
        #    raise Exception("Unrecognized WCS") from e

    return wcsobj


def correct_dark_decay(ramp_model, caldir, mylog):
    """
    Subtracts the dark decay signal from a resultant cube.

    ramp_model is updated in place.

    Parameters
    ----------
    ramp_model : roman_datamodels.datamodels.RampModel
        data model including resultant cube
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    """

    detector = ramp_model.meta.instrument.detector
    with asdf.open(caldir["dark_decay"]) as f:
        decayref = datamodels.DarkdecaysignalRefModel.create_from_model(f["roman"])
        decay_table = getattr(decayref.decay_table, detector)
        subtract_dark_decay(
            ramp_model.data,
            decay_table.amplitude,
            decay_table.time_constant,
            ramp_model.meta.exposure.frame_time,
            ramp_model.meta.exposure.read_pattern,
            int(detector[3:]),
        )
    mylog.append("Dark decay correction complete\n")
    ramp_model.meta.cal_step.dark_decay = "COMPLETE"


def correct_wfi18_transient(ramp_model, config, mylog):
    """
    Corrects the WFI18 first-read transient.

    Only applies to detector WFI18; for any other detector the model is
    returned unchanged. The ramp_model is updated in place.

    Parameters
    ----------
    ramp_model : roman_datamodels.datamodels.RampModel
        data model including resultant cube
    config : dict
        Configuration dictionary. If ``config["wfi18_mask_rows"]`` is True,
        mask the most affected rows instead of fitting and removing the anomaly.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    """

    if ramp_model.meta.instrument.detector != "WFI18":
        mylog.append("Skipping WFI18 transient correction (not WFI18)\n")
        ramp_model.meta.cal_step.wfi18_transient = "N/A"
        return
    correct_anomaly(ramp_model, mask_rows=config.get("wfi18_mask_rows", False))
    mylog.append("WFI18 transient correction complete\n")
    ramp_model.meta.cal_step.wfi18_transient = "COMPLETE"


def _embed_active(active, nb=4):
    """
    Embeds the active region into a full region that includes border pixels.

    Parameters
    ----------
    active : np.array
        2D array covering the active 4088x4088 pixels only
    nb : int
        Number of reference-pixel border rows/columns, usually 4.

    Returns
    -------
    np.array
        Full-frame array with the border zeroed and the active region
        filled. Returned as float32.

    """

    full = np.zeros(tuple(x + 2 * nb for x in active.shape), dtype=np.float32)
    full[nb:-nb, nb:-nb] = active
    return full


def do_ramp_fit(ramp_model, meta, config, caldir, mylog):
    """
    Fit a slope to a ramp model, returning a 2D rate ImageModel.

    If config["romancal_ramp_fit"] is True, use the likelihood-based
    ramp fitting approach taken in romancal; otherwise, use the specialized
    ramp fitting approach from fitting.ramp_fit.

    romancal expects to return 4088x4088 images (trimming the border pixels),
    but this pipeline expects 4096x4096 full frame images.  We work around that
    by reembedding the active region in the larger frame.  These pixels get
    trimmed off anyway in the final image.

    Parameters
    ----------
    ramp_model : roman_datamodels.datamodels.RampModel
        Ramp data model holding the linearized cube, flags, and the
        border-reference fields.
    meta : dict
        Metadata dictionary. ``meta["K"]`` (ramp weights) and
        ``meta["ramp_opt_pars"]`` are set here for downstream bookkeeping.
    config : dict
        Configuration dictionary.
    caldir : dict
        Locations of calibration files.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.

    Returns
    -------
    roman_datamodels.datamodels.ImageModel
        2D rate image (DN/s), full 4096x4096 frame.

    """

    nb = pars.nborder

    if config.get("romancal_ramp_fit", False):
        # romancal maximum-likelihood ramp fit
        with asdf.open(caldir["read"]) as fr:
            readnoise = datamodels.ReadnoiseRefModel.create_from_model(fr["roman"])
        with asdf.open(caldir["gain"]) as fg:
            gain = datamodels.GainRefModel.create_from_model(fg["roman"])
        # exclude_first handling is not required here, since these pixels
        # are marked DO_NOT_USE in the initializationstep
        image_model = ramp_fit_step.likely(
            ramp_model,
            readnoise,
            gain,
            rejection_threshold=config.get("REJECTION_THRESHOLD", 4.5),
            jump_kw=config.get("JUMP_KW", None),
        )
        meta["K"] = None  # not used by the likelihood fitter
        meta["ramp_opt_pars"] = None  # likewise
        mylog.append("romancal likelihood ramp fit complete\n")
    else:
        exclude_first = config.get("EXCLUDE_FIRST", True)
        uopt = {"slope": 0.4, "gain": 1.8, "sigma_read": 6.5}
        if "RAMP_OPT_PARS" in config:
            uopt = config["RAMP_OPT_PARS"]
        u_ = float(uopt["slope"]) / float(uopt["gain"]) / float(uopt["sigma_read"]) ** 2
        meta["K"] = fitting.construct_weights(u_, meta, exclude_first=exclude_first)
        meta["ramp_opt_pars"] = uopt
        mylog.append(f"\n\nRamp fit optimized for u = {u_:11.5E} s**-1\n")
        mylog.append("weights = {}\n".format(meta["K"]))
        if "JUMP_DETECT_PARS" in config:
            meta["jump_detect_pars"] = config["JUMP_DETECT_PARS"]
        slope, slope_err_read, slope_err_poisson = fitting.ramp_fit(
            ramp_model.data,
            ramp_model.groupdq,
            ramp_model.pixeldq,
            meta,
            caldir,
            mylog,
            exclude_first=exclude_first,
        )
        # package result into an ImageModel using a romancal routine.
        # That routine also trims the reference border and returns the
        # active region.  Ideally I want to deprecate this path and so I'm
        # using a private romancal routine for now.
        image_info = {
            "slope": slope,
            "dq": ramp_model.pixeldq,
            "err": np.hypot(slope_err_read, slope_err_poisson),
            "var_poisson": slope_err_poisson**2,
        }
        image_model = ramp_fit_step._create_image_model(ramp_model, image_info)

    # re-embed the active region into the full frame so the rest of the
    # pipeline stays full-frame. The science/variance border pixels are zeroed
    # (they are trimmed off the final product anyway); the border dq keeps the
    # reference-pixel flags carried by ramp_model.pixeldq.
    image_model.data = _embed_active(image_model.data, nb)
    image_model.var_poisson = _embed_active(image_model.var_poisson, nb)
    image_model.err = _embed_active(image_model.err, nb)
    full_dq = np.array(ramp_model.pixeldq, dtype=np.uint32)
    full_dq[nb:-nb, nb:-nb] = image_model.dq
    image_model.dq = full_dq

    return image_model


def calibrateimage(config, verbose=True):
    """
    Main routine to run the specified calibrations from a config file.

    Parameters
    ----------
    config : dict
        Configuration (likely unpacked from a YAML file).
    verbose : bool, optional
        Whether to print lots of intermediate stuff to the terminal.

    Returns
    -------
    None

    """

    # setup
    mylog = processlog.ProcessLog()

    # get an initial WCS (if provided)
    # in some simulations we may need to give this if the input stars themselves are simulated
    thewcs = wcs_from_config(config)
    caldir = config["CALDIR"]
    backup = config.get("SATURATION_BACKUP", 1)

    # initialize a data cube and data quality
    ramp_model, meta = initializationstep(config, caldir, mylog)

    nb = meta["nborder"] = pars.nborder
    mylog.append("Initialized data\n")

    # saturation check
    saturation_check(ramp_model, caldir, mylog, backup=backup)
    mylog.append("Saturation check complete\n")

    data, rdq, pdq, l1meta, amp33 = (
        ramp_model["data"],
        ramp_model["groupdq"],
        ramp_model["pixeldq"],
        ramp_model.meta,
        ramp_model["amp33"],
    )
    (ngrp, ny, nx) = np.shape(data)

    # reference pixel correction -- right now using a 5-pixel filter of the left & right ref pixels
    # and the top & bottom pixel subtraction functions from Laliotis et al. (2024)
    # **This is a placeholder until:
    #  - improved reference pixel correction from GSFC group should be available
    #
    slope = None  # will overwrite later
    with asdf.open(caldir["dark"]) as f:
        # rsub = np.zeros((ngrp, pars.nside), dtype=np.float32)
        for j in range(ngrp):
            image = np.zeros((pars.nside, pars.nside_augmented), dtype=np.float32)
            de = np.shape(f["roman"]["data"])[0] - np.shape(data)[0]
            image[:, : pars.nside] = data[j + de, :, :] - f["roman"]["data"][j + de, :, :]
            with asdf.open(caldir["read"]) as fr:
                if "amp33" in fr["roman"]:
                    image[:, -pars.channelwidth :] = amp33[j, :, :] - fr["roman"]["amp33"]["med"]
                    image[:, -pars.channelwidth :] -= np.median(image[:, -pars.channelwidth :])

                    # compute optimal slope, but only once
                    if slope is None:
                        a = fr["roman"]["amp33"]
                        cvar = fr["roman"]["anc"]["C_PINK"] ** 2
                        slope = (
                            a["M_PINK"]
                            * cvar
                            / (
                                a["M_PINK"] ** 2 * cvar
                                + a["RU_PINK"] ** 2
                                + np.median(a["std"]) ** 2 / 128 / np.log(4096)
                            )
                        )
            image = reference_subtraction.ref_subtraction_row(image, use_ref_channel=True, slope=slope)
            image = reference_subtraction.ref_subtraction_channel(image, use_ref_channel=True)
            data[j + de, :, :] = image[:, : pars.nside] + f["roman"]["data"][j + de, :, :]

    # bias correction
    if "biascorr" in caldir:
        with asdf.open(caldir["biascorr"]) as f:
            de = np.shape(f["roman"]["data"])[0] - np.shape(data)[0]
            data[:, nb:-nb, nb:-nb] -= f["roman"]["data"][de:]
        mylog.append("Included bias correction\n")
    else:
        mylog.append("Skipping bias correction\n")

    if "dark_decay" in caldir:
        correct_dark_decay(ramp_model, caldir, mylog)
    else:
        ramp_model.meta.cal_step.dark_decay = "INCOMPLETE"

    if config.get("correct_wfi18_transient", False):
        correct_wfi18_transient(ramp_model, config, mylog)
    else:
        ramp_model.meta.cal_step.wfi18_transient = "INCOMPLETE"

    # linearity correxction
    # ** right now applies the linearity to a group average, which isn't strictly correct **
    # ** will fix this in a future upgrade! **
    data, dq_lin = ipc_linearity.multilin(
        data,
        caldir["linearitylegendre"],  # the linearity cube
        do_not_flag_first=meta["read_pattern"][0]
        == [0],  # don't flag the first read for being off scale if it is the reset
        attempt_corr=~rdq
        & pixel.SATURATED,  # don't flag saturated pixels as having a bad linearity correction
    )
    pdq |= dq_lin
    del dq_lin  # we have everything we need
    mylog.append("Linearity correction complete\n")
    ramp_model.data = data  # keep data and ramp_model.data in sync

    # IPC correction
    if "ipc4d" in caldir:
        ipc_linearity.correct_cube(data, caldir["ipc4d"], mylog, gain_file=caldir["gain"])
    else:
        mylog.append("skipping IPC correction\n")

    # ramp-fit to a 2D rate ImageModel
    image_model = do_ramp_fit(ramp_model, meta, config, caldir, mylog)

    # subtract out dark current (updates image_model in place)
    subtract_dark_current(image_model, caldir, mylog)
    mylog.append("Dark current subtracted\n")

    # unpack the rate image back into full-frame arrays for use downstream
    slope = np.asarray(image_model.data, dtype=np.float32)
    pdq = np.asarray(image_model.dq, dtype=np.uint32)
    # read-noise error is derived from the total err and the Poisson variance
    # (the ramp fitter no longer exposes var_rnoise separately).
    err = np.asarray(image_model.err, dtype=np.float32)
    slope_err_poisson = np.sqrt(np.asarray(image_model.var_poisson, dtype=np.float32))
    slope_err_read = np.sqrt(np.clip(err**2 - slope_err_poisson**2, 0.0, None))

    # apply flat field
    flat = flatutils.get_flat(caldir, meta, pdq)
    # this is the ratio of the true pixel area to the reference area (0.11 arcsec)^2
    AreaFactor = (
        coordutils.pixelarea(riwcs.convert_wcs_to_gwcs(repackage_wcs(thewcs)), N=np.shape(slope)[-1])
        / pars.Omega_ideal
    )
    flat = (flat / AreaFactor).astype(np.float32)
    mylog.append("acquired flat field\n")
    for p in [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]:
        mylog.append(f" {p:2d}%ile = {np.percentile(flat, p):6.4f},")
    mylog.append("\n")
    slope /= flat
    slope_err_read /= flat
    slope_err_poisson /= flat

    # need the median gain to send to a file
    with asdf.open(caldir["gain"]) as g_:
        medgain = np.median(g_["roman"]["data"])
    mylog.append(f"median gain = {medgain:8.5f} e/DN\n")

    # blank persistence object right now
    persistence = rip.Persistence()

    # sky information
    slope_withsky = np.copy(slope)  # version before sky subtraction
    m = maskhandling.PixelMask1.build(pdq)
    medsky, _ = sky.smooth_mode(sky.binkxk(np.where(np.logical_not(m), slope, np.nan), 4))
    # if the configuration asks for simple subtraction, do it
    if "SKYORDER" in config:
        skyorder = int(config["SKYORDER"])
        skycoefs, skymodel = sky.medfit(slope[nb:-nb, nb:-nb], order=skyorder)
        slope[nb:-nb, nb:-nb] -= skymodel
        del skymodel
    else:
        skycoefs = np.array([]).astype(np.float32)
        skyorder = -1  # not used

    im2, extras2 = rimage.make_asdf(
        slope[nb:-nb, nb:-nb] * u.DN / u.s,
        (slope_err_read[nb:-nb, nb:-nb] * u.DN / u.s) ** 2,
        (slope_err_poisson[nb:-nb, nb:-nb] * u.DN / u.s) ** 2,
        metadata=l1meta,
        persistence=persistence,
        dq=pdq[nb:-nb, nb:-nb],
        imwcs=repackage_wcs(thewcs),
        gain=medgain,
    )
    # strip unit from certain fields if not needed
    for x in ["data", "var_poisson", "var_rnoise", "var_flat", "err"]:
        if x in im2 and hasattr(im2[x], "value"):
            im2[x] = im2[x].value

    # carry through the romancal ramp-fit diagnostics
    # dumo is slope-like, so flat-field it
    if config.get("romancal_ramp_fit", False):
        im2["dumo"] = (np.asarray(image_model.dumo) / flat[nb:-nb, nb:-nb]).astype(np.float16)
        im2["chisq"] = np.asarray(image_model.chisq, dtype=np.float16)

    oututils.add_in_ref_data(im2, config["IN"], rdq, pdq)

    # update the metadata
    # oututils.update_flags(im2, "gen_cal_image") # <-- this doesn't work with updated roman_datamodels,
    #                                                    but it isn't essential
    oututils.add_in_provenance(im2, "gen_cal_image")

    # process information specific to this code
    processinfo = {
        "medsky": medsky,
        "medgain": medgain,
        "skyorder": skyorder,
        "skycoefs": skycoefs,
        "ramp_opt_pars": meta["ramp_opt_pars"],
        "meta": meta,
        "weights": meta["K"],
        "config": config,
        "log": mylog.output,
        "exclude_first": config.get("EXCLUDE_FIRST", True),
    }

    # this is for getting the ramp data so we know which range was used
    # (max 127 groups)
    if "SLICEOUT" in config:
        if config["SLICEOUT"]:
            if ngrp >= 128:
                raise ValueError("too many groups")
            endslice = np.zeros((pars.nside_active, pars.nside_active), dtype=np.int8) - 1
            nb = pars.nborder
            for iend in range(1, ngrp):
                endslice = np.where(
                    rdq[iend, nb:-nb, nb:-nb] & ~rdq[iend - 1, nb:-nb, nb:-nb] & pixel.SATURATED != 0,
                    iend - 1,
                    endslice,
                )
            processinfo["endslice"] = endslice

    # Write file
    with asdf.AsdfFile() as af2:
        af2.tree = {"roman": im2, "processinfo": processinfo}
        af2.tree["roman"]["data_withsky"] = slope_withsky[nb:-nb, nb:-nb]
        if hasattr(af2.tree["roman"]["data_withsky"], "value"):
            af2.tree["roman"]["data_withsky"] = af2.tree["roman"]["data_withsky"].value
        if "cal_step" in af2.tree["roman"]["meta"]:
            print(af2.tree["roman"]["meta"]["cal_step"])
        else:
            print("cal_step not in roman->meta")
        typefix.fix(af2)
        with open(config["OUT"], "wb") as f:
            af2.write_to(f)

    if "FITSOUT" in config:
        if config["FITSOUT"]:
            good = ~maskhandling.PixelMask1.build(im2["dq"])  # this is one choice

            # note we accept saturated pixels in this step
            fits.HDUList(
                [
                    fits.PrimaryHDU(im2["data"]),
                    fits.ImageHDU(im2["dq"]),
                    fits.ImageHDU(np.where(good, im2["data"], -1000)),
                ]
            ).writeto(config["OUT"][:-5] + "_asdf_to.fits", overwrite=True)

    print(mylog.output)
    return


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    calibrateimage(config)
