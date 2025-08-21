import sys
import warnings
from copy import deepcopy

import asdf

# not actually doing a simulation but needed to pass around the WCS types
import galsim
import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits
from roman_datamodels.dqflags import pixel
from romanisim import image as rimage
from romanisim import persistence as rip
from romanisim import wcs as riwcs

# stcal imports
from stcal.saturation.saturation import flag_saturated_pixels

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
)

# local imports
from . import oututils

### function definitions below here


def wcs_from_config(config):
    """Gets a WCS object from the configuration."""

    if "FITSWCS" in config:
        with open(config["FITSWCS"]) as f:
            return fits.Header.fromstring(f.read())

    # if no WCS was found, just return None (we'll deal with this later)
    return None


def initializationstep(config, caldir, mylog):
    """Initialization step. Returns:

    data : 3D numpy array, data
    rdq : 3D numpy array, flags (ramp data quality)
    pdq : 2D numpy array, flags (pixel data quality)
    meta : dictionary of assorted other metadata
           (right now: frame_time and read_pattern)
    l1meta : metadata stright from the L1 file
    amp33 : reference output
    """

    with asdf.open(config["IN"]) as f:
        data = np.copy(f["roman"]["data"].astype(np.float32))
        amp33 = np.copy(f["roman"]["amp33"].astype(np.float32))
        rdq = np.zeros(np.shape(data), dtype=np.uint32)

        # guide windows
        # in DCL testing the rows containing the guide windows were affected
        # we expect also that the pixels with IPC coupling to the guide window
        # are affected at some level so I'm flagging those too.
        guide_star = f["roman"]["meta"]["guide_star"]
        xstart = int(guide_star["window_xstart"])
        xstop = int(guide_star["window_xstop"])
        ystart = int(guide_star["window_ystart"])
        ystop = int(guide_star["window_ystop"])
        mylog.append(f"guide window: x={xstart:d}:{xstop:d}, y={ystart:d}:{ystop:d}\n")
        # if the metadata contain a real window, mask that row
        if xstart >= 0 and ystart >= 0 and xstop <= pars.nside and ystop <= pars.nside:
            rdq[:, :, xstart:xstop] |= pixel.GW_AFFECTED_DATA
            # now flag potential IPC
            if xstart > pars.nborder:
                xstart -= 1
            if xstop < pars.nside - pars.nborder:
                xstop += 1
            if ystart > pars.nborder:
                ystart -= 1
            if ystop < pars.nside - pars.nborder:
                ystop += 1
            rdq[:, ystart:ystop, xstart:xstop] |= pixel.GW_AFFECTED_DATA

        # pull out metadata that we want later
        meta = {
            "frame_time": f["roman"]["meta"]["exposure"]["frame_time"],
            "read_pattern": f["roman"]["meta"]["exposure"]["read_pattern"],
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

        l1meta = deepcopy(f["roman"]["meta"])

    # mask
    if "mask" in caldir:
        with asdf.open(caldir["mask"]) as m:
            rdq |= m["roman"]["dq"][None, :, :]

    # pixel dq
    pdq = np.bitwise_or.reduce(rdq, axis=0)

    return data, rdq, pdq, meta, l1meta, amp33


def saturation_check(data, read_pattern, rdq, pdq, caldir, mylog):
    """Performs a saturation check on the data cube (data) using the calibration files in caldir.
    Information is appended to mylog. The flags rdq and pdq are updated in place.

    This function serves as a wrapper for flag_saturated_pixels (imported from stcal).
    """

    # passing the 0th frame will lead to division by zero, so we avoid this
    # start the saturation check with the s th frame
    s = 0
    if read_pattern[0] == [0]:
        s = 1

    with asdf.open(caldir["saturation"]) as f:
        flag_saturated_pixels(
            data[None, s:, :, :],  # flag_saturated_pixels expects a 4D array with integrations as the 0-axis
            rdq[None, s:, :, :],  # ramp data quality, with only 1 integration, expanded to 4D
            pdq,  # 2D pixel, passed through
            f["roman"]["data"],  # saturation threshold, 2D
            np.copy(f["roman"]["dq"]),  # saturation quality flags
            2**16 - 1,  # maximum of ADC output -- 16 bits
            pixel,  # this is the Roman data quality flag array
            n_pix_grow_sat=1,  # also flag 1 pixel around each saturated one
            zframe=None,
            read_pattern=read_pattern[s:],  # again, this is a list of list of ints
            bias=None,
        )

    # backs up 1 frame to be safe since if the non-linearity curve is sharp enough
    # the existing algorithm can fail on a large group
    # important to run this in ascending order
    for i in range(len(read_pattern) - 1):
        if len(read_pattern[i]) > 1:
            rdq[i, :, :] |= rdq[i + 1, :, :] & pixel.SATURATED


def subtract_dark_current(data, rdq, pdq, caldir, meta, mylog):
    """Subtracts dark current from a linearized image.

    Inputs:
    data = 3D data cube (in DN_lin, shape ngroup,4096,4096)
    rdq = 3D ramp data quanity (uint32, shape ngroup,4096,4096)
    pdq = 2D pixel data quanity (uint32, shape 4096,4096)
    caldir = calibration dictionary
    meta = metadata
    mylog = log object

    The data, rdq, and pdq are updated in place.

    Returns:
    dcsub = subtracted dark current in DN/s
    """

    with asdf.open(caldir["dark"]) as f:
        dcsub = np.copy(f["roman"]["dark_slope"])
    ngrp = meta["ngrp"]
    for j in range(ngrp):
        data[j, :, :] -= meta["tbar"][j] * dcsub
    return dcsub


def repackage_wcs(thewcs):
    """Packages a WCS to feed to romanisim."""

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
        try:
            header = fits.Header()
            thewcs.writeToFitsHeader(header, galsim.BoundsI(0, pars.nside_active, 0, pars.nside_active))
            # offset to FITS convention -- this is undone later
            header["CRPIX1"] += 1
            header["CRPIX2"] += 1
            wcsobj = Blank()
            wcsobj.header = Blank()
            wcsobj.header.header = header
            warnings.warn("Use of GalSim WCS in calibrate is not fully working yet!")
            break
        except Exception as e:
            wcsobj = None
            raise Exception("Unrecognized WCS") from e

    return wcsobj


def calibrateimage(config, verbose=True):
    """Main routine to run the specified calibrations from a config file.

    The config is a dictionary intended to be read from a YAML file, though it could also be
    written/edited here.

    """

    # setup
    mylog = processlog.ProcessLog()

    # get an initial WCS (if provided)
    # in some simulations we may need to give this if the input stars themselves are simulated
    thewcs = wcs_from_config(config)
    caldir = config["CALDIR"]

    # initialize a data cube and data quality
    data, rdq, pdq, meta, l1meta, amp33 = initializationstep(config, caldir, mylog)
    (ngrp, ny, nx) = np.shape(data)
    nb = meta["nborder"] = pars.nborder
    mylog.append("Initialized data\n")

    # saturation check
    saturation_check(data, meta["read_pattern"], rdq, pdq, caldir, mylog)
    mylog.append("Saturation check complete\n")

    # reference pixel correction -- right now using a 5-pixel filter of the left & right ref pixels
    # and the top & bottom pixel subtraction functions from Laliotis et al. (2024)
    # **This is a placeholder until:
    #  - amp33 to be implemented (currently the simulation leaves it blank)
    #  - improved reference pixel correction from GSFC group should be available
    with asdf.open(caldir["dark"]) as f:
        # rsub = np.zeros((ngrp, pars.nside), dtype=np.float32)
        for j in range(ngrp):
            image = np.zeros((pars.nside, pars.nside_augmented), dtype=np.float32)
            image[:, : pars.nside] = data[j, :, :] - f["roman"]["data"][j, :, :]
            with asdf.open(caldir["read"]) as fr:
                if "amp33" in fr["roman"]:
                    image[:, -pars.channelwidth :] = amp33[j, :, :] - fr["roman"]["amp33"]["med"]
                    image[:, -pars.channelwidth :] -= np.median(image[:, -pars.channelwidth :])
            image = reference_subtraction.ref_subtraction_row(image, use_ref_channel=True)
            image = reference_subtraction.ref_subtraction_channel(image, use_ref_channel=True)
            data[j, :, :] = image[:, : pars.nside] + f["roman"]["data"][j, :, :]

    # bias correction
    if "biascorr" in caldir:
        with asdf.open(caldir["biascorr"]) as f:
            data[:, nb:-nb, nb:-nb] -= f["roman"]["data"]
        mylog.append("Included bias correction\n")
    else:
        mylog.append("Skipping bias correction\n")

    # linearity correxction
    # ** right now applies the linearity to a group average, which isn't strictly correct **
    # ** will fix this in a future upgrade! **
    data, dq_lin = ipc_linearity.multilin(
        data,
        caldir["linearitylegendre"],  # the linearity cube
        do_not_flag_first=meta["read_pattern"][0]
        == 0,  # don't flag the first read for being off scale if it is the reset
        attempt_corr=~rdq
        & pixel.SATURATED,  # don't flag saturated pixels as having a bad linearity correction
    )
    if len(np.shape(dq_lin)) == 2:
        rdq |= dq_lin[None, :, :]
    else:
        rdq |= dq_lin
    del dq_lin  # we have everything we need
    mylog.append("Linearity correction complete\n")
    # now data is in linearized DN, floating point

    # subtract out dark current
    # dcsub is the dark current that was subtracted --- data is updated in place
    subtract_dark_current(data, rdq, pdq, caldir, meta, mylog)  # removed dcsub= assignment as it isn't used
    mylog.append("Dark current subtracted")

    # IPC correction
    if "ipc4d" in caldir:
        ipc_linearity.correct_cube(data, caldir["ipc4d"], mylog, gain_file=caldir["gain"])
    else:
        mylog.append("skipping IPC correction\n")

    # ramp fitting
    uopt = {"slope": 0.4, "gain": 1.8, "sigma_read": 6.5}
    if "RAMP_OPT_PARS" in config:
        uopt = config["RAMP_OPT_PARS"]
    u_ = float(uopt["slope"]) / float(uopt["gain"]) / float(uopt["sigma_read"]) ** 2
    meta["K"] = fitting.construct_weights(u_, meta, exclude_first=True)
    mylog.append(f"\n\nRamp fit optimized for u = {u_:11.5E} s**-1\n")
    mylog.append("weights = {}\n".format(meta["K"]))
    if "JUMP_DETECT_PARS" in config:
        meta["jump_detect_pars"] = config["JUMP_DETECT_PARS"]
    slope, slope_err_read, slope_err_poisson = fitting.ramp_fit(
        data, rdq, pdq, meta, caldir, mylog, exclude_first=True
    )

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
        mylog.append(f" {p:2d}%ile = {np.percentile(flat,p):6.4f},")
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

    oututils.add_in_ref_data(im2, config["IN"], rdq, pdq)

    # update the metadata
    oututils.update_flags(im2, "gen_cal_image")
    oututils.add_in_provenance(im2, "gen_cal_image")

    # process information specific to this code
    processinfo = {
        "medsky": medsky,
        "skyorder": skyorder,
        "skycoefs": skycoefs,
        "ramp_opt_pars": uopt,
        "meta": meta,
        "weights": meta["K"],
        "config": config,
        "log": mylog.output,
        "exclude_first": True,
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
